#!/usr/bin/env python3
"""
Training script for conditional image generation with SCAN (Self-Correction Autoregressive Network).

Modified from:
    - LLaMAGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/train/extract_codes_t2i.py
    - fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
    - nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import argparse
import math
import os
import random
import shutil
import sys
import time
from copy import deepcopy
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Add project root to path
sys.path.append("./")

from SCAN.model.generate import sample
from SCAN.model.utils import (
    calculate_num_query_tokens_for_parallel_decoding_my,
    interleave_tokens,
)
from SCAN.util import (
    instantiate_from_config,
    load_safetensors,
    save_model_safetensors,
    set_nested_key,
)
from SCAN.utils.lr_scheduler import get_scheduler
from SCAN.utils.logger import create_logger
from SCAN.utils.visualization import make_grid

from dataset.augmentation import center_crop_arr
from dataset.build import build_dataset

# Configure PyTorch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logger = get_logger(__name__)


def cycle(dataloader: DataLoader) -> torch.utils.data.DataLoader:
    """Create an infinite iterator over the dataloader."""
    while True:
        for data in dataloader:
            yield data


def generate_mask_tensor(batch_size: int, seq_len: int, zero_ratio: float) -> torch.Tensor:
    """
    Generate a mask tensor for random token masking.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        zero_ratio: Ratio of tokens to mask (set to 0)
    
    Returns:
        Mask tensor of shape (batch_size, seq_len)
    """
    ones_num = math.ceil((1 - zero_ratio) * batch_size * seq_len)
    mask_values = torch.cat([
        torch.ones(ones_num),
        torch.zeros(batch_size * seq_len - ones_num)
    ])
    
    # Shuffle for randomness
    mask_values = mask_values[torch.randperm(mask_values.size(0))]
    
    return mask_values.view(batch_size, seq_len)


def setup_accelerator(config: DictConfig, args: argparse.Namespace) -> Tuple[Accelerator, str]:
    """Setup the accelerator for distributed training."""
    exp_name = f"{args.exp_name}_bs_{config.global_batch_size}_lr_{config.optimizer.lr}"
    experiment_dir = os.path.join(args.results_dir, args.exp_name, exp_name)
    
    accelerator_config = ProjectConfiguration(project_dir=experiment_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    accelerator = Accelerator(
        project_config=accelerator_config,
        kwargs_handlers=[ddp_kwargs],
        mixed_precision=config.accelerator.mixed_precision,
        log_with=config.accelerator.log_with,
        gradient_accumulation_steps=config.accelerator.gradient_accumulation_steps,
    )
    
    return accelerator, experiment_dir


def setup_data(config: DictConfig, args: argparse.Namespace, accelerator: Accelerator) -> Tuple[DataLoader, int]:
    """Setup the data loader."""
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    dataset = build_dataset(args, transform=transform)
    
    if args.parallel:
        per_gpu_batch_size = int(
            config.global_batch_size
            // dist.get_world_size()
            // config.accelerator.gradient_accumulation_steps
        )
    else:
        per_gpu_batch_size = int(config.global_batch_size // config.accelerator.gradient_accumulation_steps)
    
    dataloader_kwargs = {
        "batch_size": per_gpu_batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": True,
    }
    
    if args.parallel:
        dataloader_kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": 8,
        })
    
    data_loader = DataLoader(dataset, **dataloader_kwargs)
    
    return data_loader, per_gpu_batch_size


def load_model_and_tokenizer(config: DictConfig, args: argparse.Namespace, accelerator: Accelerator) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Load the model and tokenizer."""
    model = instantiate_from_config(config.ar_model).to(accelerator.device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load checkpoint if provided
    if os.path.exists(args.resume):
        if args.resume.endswith('.safetensors'):
            checkpoint = load_safetensors(args.resume)
            msg = model.load_state_dict(checkpoint, strict=False)
            logger.info(f"Resume training from checkpoint: {args.resume}, {msg}")
        else:
            checkpoint = torch.load(args.resume)
            msg = model.load_state_dict(checkpoint, strict=False)
            logger.info(f"Resume training from checkpoint: {args.resume}, {msg}")
    
    # Setup tokenizer
    tokenizer = instantiate_from_config(config.tokenizer).to(accelerator.device).eval()
    ckpt = torch.load(args.vq_ckpt, map_location="cpu")
    
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    msg = tokenizer.load_state_dict(state_dict)
    logger.info(f"Load tokenizer from checkpoint: {msg}")
    
    tokenizer.eval()
    for param in tokenizer.parameters():
        param.requires_grad = False
    
    del ckpt
    
    return model, tokenizer


def create_attention_mask(
    max_seq_length: int,
    token_num_per_iter: List[int],
    revise_iter_num: int,
    token_num_each_revise: int,
    device: torch.device
) -> torch.Tensor:
    """Create attention mask for the model."""
    attn_mask = torch.zeros(max_seq_length, max_seq_length, device=device)
    
    token_list = [1]  # Start with 1 for class token
    token_list += [element for element in token_num_per_iter for _ in range(2)]
    
    mask_pointer = 0
    for token_count in token_list:
        attn_mask[mask_pointer:, mask_pointer:mask_pointer + token_count] = 1
        mask_pointer += token_count
    
    return attn_mask.unsqueeze(0).unsqueeze(0).to(torch.bool)


def prepare_batch_data(
    x: torch.Tensor,
    y: torch.Tensor,
    tokenizer: torch.nn.Module,
    config: DictConfig,
    args: argparse.Namespace,
    pre_input_flag: bool,
    pre_logits: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare batch data for training."""
    bsz = x.shape[0]
    
    if len(x.shape) == 5:  # Video data
        x = rearrange(x, 'b c n h w -> (b n) c h w')
    
    with torch.no_grad():
        args.random_ratio = random.uniform(args.random_ratio_min, args.random_ratio_max)
        
        z = tokenizer.encode_indices(x)
        z = rearrange(z, '(b l) -> b l', b=bsz)
        z_indices = z
        image_tokens = z_indices.reshape(bsz, -1)
        
        if pre_input_flag and pre_logits is not None:
            # Use previous prediction
            pre_prediction = pre_logits[:, :config.ar_model.params.block_size].max(dim=-1)[1]
            
            # Generate random tokens
            random_tensor = torch.randint(
                low=1,
                high=config.ar_model.params.vocab_size,
                size=(bsz, z_indices.shape[1]),
                device=z_indices.device
            )
            
            # Shuffle tokens
            shuffled_order = torch.empty_like(z)
            for i in range(z.size(0)):
                indices = torch.randperm(z.size(1))
                shuffled_order[i] = z[i, indices]
            
            # Combine predictions
            combine3_mask = torch.zeros((z.shape[1], 3), dtype=torch.int, device=z_indices.device)
            one_indices = torch.randint(0, 3, (z.shape[1],))
            combine3_mask[torch.arange(z.shape[1]), one_indices] = 1
            
            combined_tensor = torch.stack((pre_prediction, random_tensor, shuffled_order))
            random_tensor = torch.einsum("ibn,ni->bn", [combined_tensor.float(), combine3_mask.float()])
        else:
            args.random_ratio = 0
            random_tensor = torch.randint(
                low=1,
                high=config.ar_model.params.vocab_size,
                size=(bsz, z_indices.shape[1]),
                device=z_indices.device
            )
        
        mask = generate_mask_tensor(bsz, z_indices.shape[1], args.random_ratio).to(device=z_indices.device)
        real_n_fake_z_indices = z_indices * mask + random_tensor * (1 - mask)
        discriminator_target = torch.eq(z_indices, real_n_fake_z_indices).long()
        z_indices = real_n_fake_z_indices.long()
    
    return z_indices, image_tokens, discriminator_target


def log_training_metrics(
    accelerator: Accelerator,
    logger,
    train_steps: int,
    running_loss: float,
    running_dis_loss: float,
    running_grad_norm: float,
    start_time: float,
    data_time: float,
    log_every: int,
    lr_scheduler
) -> None:
    """Log training metrics."""
    average_loss = torch.tensor(running_loss / log_every, device=accelerator.device).item()
    average_dis_loss = torch.tensor(running_dis_loss / log_every, device=accelerator.device).item()
    average_grad_norm = torch.tensor(running_grad_norm / log_every, device=accelerator.device).item()
    
    end_time = time.time()
    average_time = (end_time - start_time) / log_every
    data_time = data_time / log_every
    
    logger.info(
        f"(Step {train_steps:08d}) "
        f"Loss {average_loss:.4f} | "
        f"Dis_loss {average_dis_loss:.4f} | "
        f"Grad Norm {average_grad_norm:.4f} | "
        f"LR {lr_scheduler.get_last_lr()[0]:.5f} | "
        f"Time {average_time:.4f}s | "
        f"Data Time {data_time:.4f}"
    )
    
    if accelerator.is_main_process:
        logger_dict = {
            "loss": average_loss,
            "benchmark/time": average_time,
            "grad_norm": average_grad_norm,
            "lr": lr_scheduler.get_last_lr()[0]
        }
        accelerator.log(logger_dict, step=train_steps)


def visualize_samples(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: torch.nn.Module,
    logits: torch.Tensor,
    cond: torch.Tensor,
    image_tokens: torch.Tensor,
    token_order: torch.Tensor,
    args: argparse.Namespace,
    config: DictConfig,
    experiment_dir: str,
    train_steps: int
) -> None:
    """Generate and save visualization samples."""
    with torch.no_grad():
        img_token_num = config.ar_model.params.block_size
        visualize_num = min(args.visualize_num, logits.shape[0])
        
        visualize_logits = logits[:visualize_num, :img_token_num]
        visualize_cond = cond[:visualize_num]
        visualize_token_order = token_order[:visualize_num, :img_token_num]
        visualize_gt_indices = image_tokens[:visualize_num]
        orig_token_order = torch.argsort(visualize_token_order)
        
        # Teacher forcing reconstruction
        pred_recon_indices = torch.zeros(visualize_num, img_token_num, device=accelerator.device).long()
        for i in range(img_token_num):
            pred_recon_indices[:, i:i+1] = torch.argmax(visualize_logits[:, i:i+1], dim=-1)
        
        pred_recon_indices = torch.gather(
            pred_recon_indices.unsqueeze(-1),
            dim=1,
            index=orig_token_order.unsqueeze(-1)
        ).squeeze(-1)
        pred_recon_imgs = tokenizer.decode_codes_to_img(pred_recon_indices, args.image_size)
        
        # Ground truth reconstruction
        gt_recon_imgs = tokenizer.decode_codes_to_img(visualize_gt_indices, args.image_size)
        
        # Generation
        revise_token_nums = [args.token_num_each_revise] * args.revise_iter_num
        max_seq_length = (
            config.ar_model.params.cls_token_num +
            config.ar_model.params.block_size * 2 +
            args.revise_iter_num * args.token_num_each_revise * 2
        )
        
        gen_mask = torch.zeros(
            max_seq_length - args.revise_iter_num * args.token_num_each_revise * 2,
            max_seq_length - args.revise_iter_num * args.token_num_each_revise * 2,
            device=accelerator.device
        )
        
        token_num_per_iter = calculate_num_query_tokens_for_parallel_decoding_my(
            args.num_inference_steps,
            config.ar_model.params.block_size
        )
        
        start_time = time.time()
        
        if args.parallel:
            gen_indices, before_revised_outputs = model.module.generate_embs_then_tokens(
                cond=visualize_cond,
                token_order=None,
                cfg_scales=[4.0, 4.0],
                num_inference_steps=args.num_inference_steps,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                token_num_per_iter=token_num_per_iter[:-args.revise_iter_num],
                mask=gen_mask,
                revise_token_nums=revise_token_nums,
            )
            model.module.remove_caches()
        else:
            gen_indices, before_revised_outputs = model.generate_embs_then_tokens(
                cond=visualize_cond,
                token_order=None,
                cfg_scales=[4.0, 4.0],
                num_inference_steps=args.num_inference_steps,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                token_num_per_iter=token_num_per_iter[:-args.revise_iter_num],
                mask=gen_mask,
                revise_token_nums=revise_token_nums,
            )
            model.remove_caches()
        
        print(f"Sampling takes about {time.time() - start_time:.2f} seconds.")
        
        gen_imgs = tokenizer.decode_codes_to_img(gen_indices, args.image_size)
        before_revised_outputs = tokenizer.decode_codes_to_img(before_revised_outputs, args.image_size)
        
        # Create visualization grid
        out_image = np.concatenate(
            (gen_imgs, before_revised_outputs, pred_recon_imgs, gt_recon_imgs),
            axis=0
        )
        out_image = make_grid(
            out_image,
            num_row=4 * max(gen_imgs.shape[0] // 8, 1),
            scale=1
        )
        
        samples_path = os.path.join(experiment_dir, 'samples')
        os.makedirs(samples_path, exist_ok=True)
        
        im = Image.fromarray(out_image)
        im.save(os.path.join(samples_path, f"sample_trainSteps_{train_steps}.png"))


def save_checkpoint(accelerator: Accelerator, checkpoint_dir: str, train_steps: int, keep_last_k: int, ckpt_every: int) -> None:
    """Save model checkpoint."""
    ckpt_path = os.path.join(checkpoint_dir, f"iters_{train_steps:08d}")
    os.makedirs(ckpt_path, exist_ok=True)
    accelerator.save_state(ckpt_path)
    logger.info(f"Saved Iter {train_steps} checkpoint to {ckpt_path}")
    
    # Clean up old checkpoints
    for ckpt_dir in os.listdir(checkpoint_dir):
        if ckpt_dir.startswith("iters") and ckpt_dir != f"iters_{train_steps:08d}":
            shutil.rmtree(os.path.join(checkpoint_dir, ckpt_dir))


def main(args: argparse.Namespace) -> None:
    """Main training function."""
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Load configuration
    config = OmegaConf.load(args.config)
    config.global_batch_size = args.global_batch_size
    assert args.visualize_num <= args.global_batch_size
    
    # Setup accelerator
    accelerator, experiment_dir = setup_accelerator(config, args)
    set_seed(config.global_seed + accelerator.process_index)
    
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    
    if accelerator.is_main_process:
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger_main = create_logger(experiment_dir, args.parallel)
        logger_main.info(f"Experiment directory: {experiment_dir}")
        logger_main.info(f"Checkpoint directory: {checkpoint_dir}")
        logger_main.info(accelerator.state)
    else:
        logger_main = create_logger(None)
    
    # Setup data and model
    data_loader, per_gpu_batch_size = setup_data(config, args, accelerator)
    model, tokenizer = load_model_and_tokenizer(config, args, accelerator)
    
    # Setup optimizer and scheduler
    optimizer = model.configure_optimizer(**config.optimizer)
    lr_scheduler = get_scheduler(
        name=config.lr_scheduler.type,
        optimizer=optimizer,
        num_warmup_steps=config.lr_scheduler.warm_up_iters * 
                        config.accelerator.gradient_accumulation_steps * accelerator.num_processes,
        num_training_steps=config.max_iters * 
                          config.accelerator.gradient_accumulation_steps * accelerator.num_processes,
        min_lr_ratio=config.lr_scheduler.min_lr_ratio,
        num_cycles=config.lr_scheduler.num_cycles,
    )
    
    # Prepare for distributed training
    model.train()
    model, optimizer, data_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, data_loader, lr_scheduler
    )
    data_loader = cycle(data_loader)
    
    # Resume training if checkpoint exists
    train_steps = 0
    if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
        saved_ckpt_dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("iters")]
        saved_ckpt_dirs = sorted(saved_ckpt_dirs)
        ckpt_dir = os.path.join(checkpoint_dir, saved_ckpt_dirs[-1])
        
        if accelerator.is_main_process:
            logger_main.info(f"Resuming from {ckpt_dir}")
        
        accelerator.load_state(ckpt_dir)
        train_steps = int(saved_ckpt_dirs[-1].split("_")[-1])
    
    # Setup attention mask
    token_num_per_iter = calculate_num_query_tokens_for_parallel_decoding_my(
        args.num_inference_steps,
        config.ar_model.params.block_size
    )
    
    max_seq_length = (
        config.ar_model.params.cls_token_num +
        config.ar_model.params.block_size * 2 +
        args.revise_iter_num * args.token_num_each_revise * 2
    )
    
    for _ in range(args.revise_iter_num):
        token_num_per_iter.append(args.token_num_each_revise)
    
    attn_mask = create_attention_mask(
        max_seq_length, token_num_per_iter, args.revise_iter_num,
        args.token_num_each_revise, accelerator.device
    )
    
    # Training loop
    model.train()
    logger_main.info(f"Starting training from iteration {train_steps} to {config.max_iters}")
    
    log_iters, running_loss, running_grad_norm, running_dis_loss = 0, 0, 0, 0
    start_time, data_time = time.time(), 0
    pre_input_flag, pre_logits = False, None
    
    while train_steps < config.max_iters:
        model.train()
        
        if not pre_input_flag:
            data_time -= time.time()
            x, y = next(data_loader)
            x, y = x.to(accelerator.device, non_blocking=True), y.to(accelerator.device, non_blocking=True)
            cond = y.reshape(-1)
            data_time += time.time()
        
        # Prepare batch data
        z_indices, image_tokens, discriminator_target = prepare_batch_data(
            x, y, tokenizer, config, args, pre_input_flag, pre_logits
        )
        
        # Training step
        with accelerator.accumulate(model):
            logits, loss, token_order, dis_loss = model(
                z_indices, cond, targets=image_tokens,
                token_num_per_iter=token_num_per_iter,
                mask=attn_mask.repeat(x.shape[0], 1, 1, 1),
                discriminator_target=discriminator_target,
                revise_iter_num=args.revise_iter_num
            )
            
            loss += dis_loss
            accelerator.backward(loss)
            
            if config.optimizer.max_grad_norm != 0.0:
                accelerator.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
            
            # Calculate gradient norm
            grad_norm = sum(p.grad.data.norm(2).item() for p in model.parameters())
            
            # Update model
            if grad_norm < config.optimizer.skip_grad_norm or train_steps < config.optimizer.skip_grad_iter:
                optimizer.step()
            
            optimizer.zero_grad()
            lr_scheduler.step()
            
            # Update running metrics
            running_loss += (
                accelerator.gather(loss.repeat(per_gpu_batch_size)).mean().item() /
                config.accelerator.gradient_accumulation_steps
            )
            running_grad_norm += (
                grad_norm / config.accelerator.gradient_accumulation_steps
            )
            running_dis_loss += (
                accelerator.gather(dis_loss.repeat(per_gpu_batch_size)).mean().item() /
                config.accelerator.gradient_accumulation_steps
            )
            
            pre_logits = logits.detach()
            pre_input_flag = not pre_input_flag
        
        if accelerator.sync_gradients:
            log_iters += 1
            train_steps += 1
            
            # Logging
            if train_steps % args.log_every == 0 and accelerator.is_main_process:
                log_training_metrics(
                    accelerator, logger_main, train_steps, running_loss,
                    running_dis_loss, running_grad_norm, start_time,
                    data_time, args.log_every, lr_scheduler
                )
                
                running_loss, running_dis_loss, running_grad_norm = 0, 0, 0
                data_time, start_time = 0, time.time()
            
            # Visualization
            if train_steps % args.visualize_every == 0 and accelerator.is_main_process:
                model.eval()
                visualize_samples(
                    accelerator, model, tokenizer, logits, cond, image_tokens,
                    token_order, args, config, experiment_dir, train_steps
                )
                model.train()
            
            # Checkpoint saving
            if train_steps % args.ckpt_every == 0 and accelerator.is_main_process:
                save_checkpoint(accelerator, checkpoint_dir, train_steps, args.keep_last_k, args.ckpt_every)
            
            accelerator.wait_for_everyone()
    
    # Save final checkpoint
    if accelerator.is_main_process:
        final_ckpt_dir = os.path.join(checkpoint_dir, f"iters_{train_steps:08d}_final")
        os.makedirs(final_ckpt_dir, exist_ok=True)
        accelerator.save_state(final_ckpt_dir)
        logger_main.info(f"Saved Final Iter {train_steps} checkpoint to {final_ckpt_dir}")
    
    accelerator.wait_for_everyone()
    logger_main.info("Training Done.")
    accelerator.end_training()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SCAN for conditional image generation")
    
    # Model and training configuration
    parser.add_argument("--config", type=str, default="configs/SCAN/randmaskar_xl_0.7b_llamagen_dis.yaml")
    parser.add_argument("--exp-name", type=str, default='test')
    parser.add_argument("--gpt-ckpt", type=str, default='', help="ckpt path for resume training")
    parser.add_argument("--ema", action="store_true", help="whether using ema training")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--results-dir", type=str, default="results")
    
    # Model parameters
    parser.add_argument("--image-size", type=int, choices=[128, 256, 384, 448, 512], default=512)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-inference-steps", type=int, default=160)
    
    # Training parameters
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--max-iters", type=int, default=100000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    
    # Optimization parameters
    parser.add_argument("--random-ratio", type=float, default=0.5)
    parser.add_argument("--random-ratio-max", type=float, default=0.1)
    parser.add_argument("--random-ratio-min", type=float, default=0.5)
    parser.add_argument("--revise-iter-num", type=int, default=20)
    parser.add_argument("--token-num-each-revise", type=int, default=4)
    
    # Checkpoint and logging
    parser.add_argument("--resume", type=str, default='model.safetensors')
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--keep-last-k", type=int, default=2)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--visualize-every", type=int, default=5000)
    parser.add_argument("--visualize-num", type=int, default=8)
    
    # Model checkpoints
    parser.add_argument("--vq-ckpt", type=str, default='/new_shanghai/fengzheng.fzz/models/LlamaGen/vq_ds16_c2i.pt')
    
    # Data parameters
    parser.add_argument("--dataset", type=str, choices=['imagenetraw', 't2i', 'ucf', 'pandas'], default='imagenetraw')
    parser.add_argument("--data-path", type=str, default='/datacube_nas/datasets/labels/dingsunbao/qiyin.hqy/imagenet/train')
    parser.add_argument("--data-anno", type=str, default='/new_shanghai/fengzheng.fzz/data/data_configs/ImageNet_Train_paths_mainStation.txt')
    parser.add_argument("--imagenet-class", type=str, default='/datacube_nas/datasets/labels/dingsunbao/qiyin.hqy/imagenet/labels.txt')
    
    # Additional data parameters
    parser.add_argument('--nframes', default=1, type=int)
    parser.add_argument('--return-idx', type=bool, default=False)
    parser.add_argument('--return-class-text', type=bool, default=False)
    
    # Distributed training
    parser.add_argument("--parallel", action="store_true", default=False)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)