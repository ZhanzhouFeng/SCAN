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
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
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

# Constants
DEFAULT_CONFIG = "configs/SCAN/randmaskar_xl_0.7b_llamagen_dis.yaml"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_VQ_CKPT = "/new_shanghai/fengzheng.fzz/models/LlamaGen/vq_ds16_c2i.pt"


def create_infinite_dataloader(dataloader: DataLoader) -> torch.utils.data.DataLoader:
    """Create an infinite iterator over the dataloader."""
    while True:
        for data in dataloader:
            yield data


def generate_random_mask(
    batch_size: int, 
    sequence_length: int, 
    mask_ratio: float
) -> torch.Tensor:
    """Generate a random mask tensor for token masking.
    
    Args:
        batch_size: Batch size
        sequence_length: Length of the sequence
        mask_ratio: Ratio of tokens to mask (0.0 to 1.0)
        
    Returns:
        Mask tensor of shape (batch_size, sequence_length)
    """
    total_tokens = batch_size * sequence_length
    ones_count = math.ceil((1 - mask_ratio) * total_tokens)
    
    mask_values = torch.cat([
        torch.ones(ones_count),
        torch.zeros(total_tokens - ones_count)
    ])
    
    # Random shuffle for uniform masking
    mask_values = mask_values[torch.randperm(mask_values.size(0))]
    
    return mask_values.view(batch_size, sequence_length)


def setup_training_accelerator(
    config: DictConfig, 
    args: argparse.Namespace
) -> Tuple[Accelerator, str]:
    """Initialize the accelerator for distributed training."""
    experiment_name = f"{args.exp_name}_bs_{config.global_batch_size}_lr_{config.optimizer.lr}"
    experiment_dir = os.path.join(args.results_dir, args.exp_name, experiment_name)
    
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


def setup_training_data(
    config: DictConfig, 
    args: argparse.Namespace, 
    accelerator: Accelerator
) -> Tuple[DataLoader, int]:
    """Setup the training data loader."""
    transform = transforms.Compose([
        transforms.Lambda(
            lambda pil_image: center_crop_arr(pil_image, args.image_size)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5], 
            std=[0.5, 0.5, 0.5], 
            inplace=True
        )
    ])
    
    dataset = build_dataset(args, transform=transform)
    
    # Calculate batch size per GPU
    if args.parallel:
        world_size = dist.get_world_size()
        per_gpu_batch_size = config.global_batch_size // world_size // config.accelerator.gradient_accumulation_steps
    else:
        per_gpu_batch_size = config.global_batch_size // config.accelerator.gradient_accumulation_steps
    
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


def load_model_components(
    config: DictConfig, 
    args: argparse.Namespace, 
    accelerator: Accelerator
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Load the model and tokenizer components."""
    model = instantiate_from_config(config.ar_model).to(accelerator.device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"GPT Parameters: {total_params:,}")
    
    # Load checkpoint if provided
    if os.path.exists(args.resume):
        if args.resume.endswith('.safetensors'):
            checkpoint = load_safetensors(args.resume)
            load_result = model.load_state_dict(checkpoint, strict=False)
            logger.info(f"Resumed training from: {args.resume}, {load_result}")
        else:
            checkpoint = torch.load(args.resume)
            load_result = model.load_state_dict(checkpoint, strict=False)
            logger.info(f"Resumed training from: {args.resume}, {load_result}")
    
    # Initialize tokenizer
    tokenizer = instantiate_from_config(config.tokenizer).to(accelerator.device).eval()
    ckpt = torch.load(args.vq_ckpt, map_location="cpu")
    
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    load_result = tokenizer.load_state_dict(state_dict)
    logger.info(f"Loaded tokenizer from checkpoint: {load_result}")
    
    # Freeze tokenizer parameters
    tokenizer.eval()
    for param in tokenizer.parameters():
        param.requires_grad = False
    
    del ckpt
    
    return model, tokenizer


def create_attention_mask_for_training(
    max_seq_length: int,
    token_counts_per_iteration: List[int],
    revision_iterations: int,
    tokens_per_revision: int,
    device: torch.device
) -> torch.Tensor:
    """Create attention mask for training with revision mechanism."""
    attention_mask = torch.zeros(max_seq_length, max_seq_length, device=device)
    
    # Build token sequence pattern
    token_pattern = [1]  # Start with class token
    token_pattern += [count for count in token_counts_per_iteration for _ in range(2)]
    
    # Create causal attention mask
    mask_position = 0
    for token_count in token_pattern:
        attention_mask[mask_position:, mask_position:mask_position + token_count] = 1
        mask_position += token_count
    
    return attention_mask.unsqueeze(0).unsqueeze(0).to(torch.bool)


def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    tokenizer: torch.nn.Module,
    config: DictConfig,
    args: argparse.Namespace,
    use_previous_prediction: bool,
    previous_logits: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare a batch of data for training."""
    batch_size = images.shape[0]
    
    # Handle video data format
    if len(images.shape) == 5:  # (b, c, n, h, w) -> (b*n, c, h, w)
        images = rearrange(images, 'b c n h w -> (b n) c h w')
    
    with torch.no_grad():
        # Generate random masking ratio
        args.random_ratio = random.uniform(args.random_ratio_min, args.random_ratio_max)
        
        # Encode images to tokens
        encoded_tokens = tokenizer.encode_indices(images)
        encoded_tokens = rearrange(encoded_tokens, '(b l) -> b l', b=batch_size)
        image_tokens = encoded_tokens.reshape(batch_size, -1)
        
        if use_previous_prediction and previous_logits is not None:
            # Use previous model prediction
            previous_prediction = previous_logits[:, :config.ar_model.params.block_size].argmax(dim=-1)
            
            # Generate random tokens for masking
            random_tokens = torch.randint(
                low=1,
                high=config.ar_model.params.vocab_size,
                size=(batch_size, encoded_tokens.shape[1]),
                device=encoded_tokens.device
            )
            
            # Create shuffled version of ground truth
            shuffled_tokens = torch.empty_like(encoded_tokens)
            for i in range(encoded_tokens.size(0)):
                indices = torch.randperm(encoded_tokens.size(1))
                shuffled_tokens[i] = encoded_tokens[i, indices]
            
            # Combine predictions using random weights
            combination_weights = torch.zeros(
                (encoded_tokens.shape[1], 3), 
                dtype=torch.int, 
                device=encoded_tokens.device
            )
            selected_indices = torch.randint(0, 3, (encoded_tokens.shape[1],))
            combination_weights[torch.arange(encoded_tokens.shape[1]), selected_indices] = 1
            
            combined_tokens = torch.stack([previous_prediction, random_tokens, shuffled_tokens])
            masked_tokens = torch.einsum(
                "ibn,ni->bn", 
                [combined_tokens.float(), combination_weights.float()]
            )
        else:
            # No previous prediction, use pure random masking
            args.random_ratio = 0
            masked_tokens = torch.randint(
                low=1,
                high=config.ar_model.params.vocab_size,
                size=(batch_size, encoded_tokens.shape[1]),
                device=encoded_tokens.device
            )
        
        # Apply masking
        mask = generate_random_mask(
            batch_size, 
            encoded_tokens.shape[1], 
            args.random_ratio
        ).to(device=encoded_tokens.device)
        
        final_tokens = encoded_tokens * mask + masked_tokens * (1 - mask)
        discriminator_targets = (encoded_tokens == final_tokens).long()
        
    return final_tokens.long(), image_tokens, discriminator_targets, final_tokens


def log_training_progress(
    accelerator: Accelerator,
    logger,
    current_step: int,
    accumulated_loss: float,
    accumulated_dis_loss: float,
    accumulated_grad_norm: float,
    start_time: float,
    data_loading_time: float,
    log_frequency: int,
    learning_rate_scheduler
) -> None:
    """Log training progress and metrics."""
    avg_loss = torch.tensor(accumulated_loss / log_frequency, device=accelerator.device).item()
    avg_dis_loss = torch.tensor(accumulated_dis_loss / log_frequency, device=accelerator.device).item()
    avg_grad_norm = torch.tensor(accumulated_grad_norm / log_frequency, device=accelerator.device).item()
    
    end_time = time.time()
    avg_time_per_step = (end_time - start_time) / log_frequency
    avg_data_time = data_loading_time / log_frequency
    
    logger.info(
        f"(Step {current_step:08d}) "
        f"Loss: {avg_loss:.4f} | "
        f"Dis Loss: {avg_dis_loss:.4f} | "
        f"Grad Norm: {avg_grad_norm:.4f} | "
        f"LR: {learning_rate_scheduler.get_last_lr()[0]:.6f} | "
        f"Time: {avg_time_per_step:.3f}s | "
        f"Data Time: {avg_data_time:.3f}s"
    )
    
    if accelerator.is_main_process:
        log_dict = {
            "train/loss": avg_loss,
            "train/discriminator_loss": avg_dis_loss,
            "train/gradient_norm": avg_grad_norm,
            "train/learning_rate": learning_rate_scheduler.get_last_lr()[0],
            "train/time_per_step": avg_time_per_step,
            "train/data_loading_time": avg_data_time,
        }
        accelerator.log(log_dict, step=current_step)


def generate_visualization_samples(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: torch.nn.Module,
    logits: torch.Tensor,
    conditions: torch.Tensor,
    ground_truth_tokens: torch.Tensor,
    token_order: torch.Tensor,
    args: argparse.Namespace,
    config: DictConfig,
    experiment_dir: str,
    current_step: int
) -> None:
    """Generate and save visualization samples during training."""
    with torch.no_grad():
        image_token_count = config.ar_model.params.block_size
        sample_count = min(args.visualize_num, logits.shape[0])
        
        # Extract samples for visualization
        sample_logits = logits[:sample_count, :image_token_count]
        sample_conditions = conditions[:sample_count]
        sample_token_order = token_order[:sample_count, :image_token_count]
        sample_gt_tokens = ground_truth_tokens[:sample_count]
        
        # Reconstruct original token order
        original_order = torch.argsort(sample_token_order)
        
        # Teacher forcing reconstruction
        reconstructed_tokens = sample_logits.argmax(dim=-1)
        reconstructed_tokens = torch.gather(
            reconstructed_tokens.unsqueeze(-1),
            dim=1,
            index=original_order.unsqueeze(-1)
        ).squeeze(-1)
        
        reconstructed_images = tokenizer.decode_codes_to_img(
            reconstructed_tokens, 
            args.image_size
        )
        
        # Ground truth reconstruction
        gt_images = tokenizer.decode_codes_to_img(
            sample_gt_tokens, 
            args.image_size
        )
        
        # Generate new samples
        revision_tokens = [args.token_num_each_revise] * args.revise_iter_num
        max_sequence_length = (
            config.ar_model.params.cls_token_num +
            config.ar_model.params.block_size * 2 +
            args.revise_iter_num * args.token_num_each_revise * 2
        )
        
        generation_mask = torch.zeros(
            max_sequence_length - args.revise_iter_num * args.token_num_each_revise * 2,
            max_sequence_length - args.revise_iter_num * args.token_num_each_revise * 2,
            device=accelerator.device
        )
        
        token_counts = calculate_num_query_tokens_for_parallel_decoding_my(
            args.num_inference_steps,
            config.ar_model.params.block_size
        )
        
        # Generate samples
        start_time = time.time()
        
        if args.parallel:
            generated_tokens, pre_revision_tokens = model.module.generate_embs_then_tokens(
                cond=sample_conditions,
                token_order=None,
                cfg_scales=[4.0, 4.0],
                num_inference_steps=args.num_inference_steps,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                token_num_per_iter=token_counts[:-args.revise_iter_num],
                mask=generation_mask,
                revise_token_nums=revision_tokens,
            )
            model.module.remove_caches()
        else:
            generated_tokens, pre_revision_tokens = model.generate_embs_then_tokens(
                cond=sample_conditions,
                token_order=None,
                cfg_scales=[4.0, 4.0],
                num_inference_steps=args.num_inference_steps,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                token_num_per_iter=token_counts[:-args.revise_iter_num],
                mask=generation_mask,
                revise_token_nums=revision_tokens,
            )
            model.remove_caches()
        
        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds")
        
        # Decode generated images
        generated_images = tokenizer.decode_codes_to_img(generated_tokens, args.image_size)
        pre_revision_images = tokenizer.decode_codes_to_img(pre_revision_tokens, args.image_size)
        
        # Create visualization grid
        visualization_images = np.concatenate([
            generated_images,
            pre_revision_images,
            reconstructed_images,
            gt_images
        ], axis=0)
        
        grid_image = make_grid(
            visualization_images,
            num_row=4 * max(generated_images.shape[0] // 8, 1),
            scale=1
        )
        
        # Save visualization
        samples_dir = os.path.join(experiment_dir, 'samples')
        os.makedirs(samples_dir, exist_ok=True)
        
        image = Image.fromarray(grid_image)
        image.save(os.path.join(samples_dir, f"sample_step_{current_step:08d}.png"))


def save_training_checkpoint(
    accelerator: Accelerator, 
    checkpoint_dir: str, 
    current_step: int, 
    keep_last_k: int, 
    checkpoint_frequency: int
) -> None:
    """Save model checkpoint during training."""
    checkpoint_path = os.path.join(checkpoint_dir, f"step_{current_step:08d}")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    accelerator.save_state(checkpoint_path)
    logger.info(f"Saved checkpoint at step {current_step} to {checkpoint_path}")
    
    # Clean up old checkpoints
    existing_checkpoints = [
        d for d in os.listdir(checkpoint_dir) 
        if d.startswith("step_") and d != f"step_{current_step:08d}"
    ]
    
    if len(existing_checkpoints) > keep_last_k:
        # Sort by step number and remove oldest
        existing_checkpoints.sort(key=lambda x: int(x.split('_')[1]))
        for old_checkpoint in existing_checkpoints[:-keep_last_k]:
            shutil.rmtree(os.path.join(checkpoint_dir, old_checkpoint))


def main(args: argparse.Namespace) -> None:
    """Main training function for SCAN conditional image generation."""
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires at least one CUDA-capable GPU")
    
    # Load configuration
    config = OmegaConf.load(args.config)
    config.global_batch_size = args.global_batch_size
    
    if args.visualize_num > args.global_batch_size:
        logger.warning(
            f"Visualization samples ({args.visualize_num}) exceeds batch size "
            f"({args.global_batch_size}), reducing to batch size"
        )
        args.visualize_num = args.global_batch_size
    
    # Setup accelerator and directories
    accelerator, experiment_dir = setup_training_accelerator(config, args)
    set_seed(config.global_seed + accelerator.process_index)
    
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    
    if accelerator.is_main_process:
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        main_logger = create_logger(experiment_dir, args.parallel)
        main_logger.info(f"Experiment directory: {experiment_dir}")
        main_logger.info(f"Checkpoint directory: {checkpoint_dir}")
        main_logger.info(f"Accelerator state: {accelerator.state}")
    else:
        main_logger = create_logger(None)
    
    # Setup training components
    data_loader, per_gpu_batch_size = setup_training_data(config, args, accelerator)
    model, tokenizer = load_model_components(config, args, accelerator)
    
    # Setup optimizer and learning rate scheduler
    optimizer = model.configure_optimizer(**config.optimizer)
    
    total_training_steps = config.max_iters * config.accelerator.gradient_accumulation_steps * accelerator.num_processes
    warmup_steps = config.lr_scheduler.warm_up_iters * config.accelerator.gradient_accumulation_steps * accelerator.num_processes
    
    lr_scheduler = get_scheduler(
        name=config.lr_scheduler.type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
        min_lr_ratio=config.lr_scheduler.min_lr_ratio,
        num_cycles=config.lr_scheduler.num_cycles,
    )
    
    # Prepare for distributed training
    model.train()
    model, optimizer, data_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, data_loader, lr_scheduler
    )
    data_loader = create_infinite_dataloader(data_loader)
    
    # Resume training from checkpoint if available
    current_step = 0
    if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
        checkpoint_dirs = [
            d for d in os.listdir(checkpoint_dir) 
            if d.startswith("step_")
        ]
        checkpoint_dirs.sort(key=lambda x: int(x.split('_')[1]))
        
        if checkpoint_dirs:
            latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_dirs[-1])
            
            if accelerator.is_main_process:
                main_logger.info(f"Resuming training from: {latest_checkpoint}")
            
            accelerator.load_state(latest_checkpoint)
            current_step = int(checkpoint_dirs[-1].split('_')[1])
    
    # Setup attention mask for training
    token_counts = calculate_num_query_tokens_for_parallel_decoding_my(
        args.num_inference_steps,
        config.ar_model.params.block_size
    )
    
    max_sequence_length = (
        config.ar_model.params.cls_token_num +
        config.ar_model.params.block_size * 2 +
        args.revise_iter_num * args.token_num_each_revise * 2
    )
    
    # Add revision tokens to token counts
    token_counts.extend([args.token_num_each_revise] * args.revise_iter_num)
    
    attention_mask = create_attention_mask_for_training(
        max_sequence_length, 
        token_counts[:-args.revise_iter_num], 
        args.revise_iter_num,
        args.token_num_each_revise, 
        accelerator.device
    )
    
    # Training loop
    model.train()
    main_logger.info(f"Starting training from step {current_step} to {config.max_iters}")
    
    # Training metrics
    accumulated_loss = 0.0
    accumulated_dis_loss = 0.0
    accumulated_grad_norm = 0.0
    log_iterations = 0
    
    start_time = time.time()
    data_loading_time = 0.0
    
    use_previous_prediction = False
    previous_logits = None
    
    # Initialize training variables
    images = None
    labels = None
    conditions = None
    
    while current_step < config.max_iters:
        model.train()
        
        if not use_previous_prediction:
            data_loading_time -= time.time()
            images, labels = next(data_loader)
            images = images.to(accelerator.device, non_blocking=True)
            labels = labels.to(accelerator.device, non_blocking=True)
            conditions = labels.reshape(-1)
            data_loading_time += time.time()
        
        # Prepare batch data
        token_indices, image_tokens, discriminator_targets, _ = prepare_training_batch(
            images, labels, tokenizer, config, args, 
            use_previous_prediction, previous_logits
        )
        
        # Training step
        with accelerator.accumulate(model):
            logits, loss, token_order, dis_loss = model(
                token_indices,
                conditions,
                targets=image_tokens,
                token_num_per_iter=token_counts,
                mask=attention_mask.repeat(images.shape[0], 1, 1, 1),
                discriminator_target=discriminator_targets,
                revise_iter_num=args.revise_iter_num
            )
            
            total_loss = loss + dis_loss
            accelerator.backward(total_loss)
            
            # Gradient clipping
            if config.optimizer.max_grad_norm > 0.0:
                accelerator.clip_grad_norm_(
                    model.parameters(), 
                    config.optimizer.max_grad_norm
                )
            
            # Calculate gradient norm
            grad_norm = sum(
                p.grad.data.norm(2).item() 
                for p in model.parameters() 
                if p.grad is not None
            )
            
            # Update model parameters
            should_skip_update = (
                grad_norm >= config.optimizer.skip_grad_norm and 
                current_step >= config.optimizer.skip_grad_iter
            )
            
            if not should_skip_update:
                optimizer.step()
            
            optimizer.zero_grad()
            lr_scheduler.step()
            
            # Update metrics
            loss_value = accelerator.gather(
                loss.repeat(per_gpu_batch_size)
            ).mean().item() / config.accelerator.gradient_accumulation_steps
            
            dis_loss_value = accelerator.gather(
                dis_loss.repeat(per_gpu_batch_size)
            ).mean().item() / config.accelerator.gradient_accumulation_steps
            
            accumulated_loss += loss_value
            accumulated_dis_loss += dis_loss_value
            accumulated_grad_norm += grad_norm / config.accelerator.gradient_accumulation_steps
            
            previous_logits = logits.detach()
            use_previous_prediction = not use_previous_prediction
        
        if accelerator.sync_gradients:
            log_iterations += 1
            current_step += 1
            
            # Logging
            if current_step % args.log_every == 0 and accelerator.is_main_process:
                log_training_progress(
                    accelerator, main_logger, current_step,
                    accumulated_loss, accumulated_dis_loss, 
                    accumulated_grad_norm, start_time, 
                    data_loading_time, args.log_every, lr_scheduler
                )
                
                # Reset metrics
                accumulated_loss = 0.0
                accumulated_dis_loss = 0.0
                accumulated_grad_norm = 0.0
                data_loading_time = 0.0
                start_time = time.time()
            
            # Visualization
            if current_step % args.visualize_every == 0 and accelerator.is_main_process:
                model.eval()
                generate_visualization_samples(
                    accelerator, model, tokenizer, logits, conditions,
                    image_tokens, token_order, args, config, 
                    experiment_dir, current_step
                )
                model.train()
            
            # Checkpoint saving
            if current_step % args.ckpt_every == 0 and accelerator.is_main_process:
                save_training_checkpoint(
                    accelerator, checkpoint_dir, current_step, 
                    args.keep_last_k, args.ckpt_every
                )
            
            accelerator.wait_for_everyone()
    
    # Save final checkpoint
    if accelerator.is_main_process:
        final_checkpoint = os.path.join(checkpoint_dir, f"step_{current_step:08d}_final")
        os.makedirs(final_checkpoint, exist_ok=True)
        accelerator.save_state(final_checkpoint)
        main_logger.info(f"Saved final checkpoint at step {current_step} to {final_checkpoint}")
    
    accelerator.wait_for_everyone()
    main_logger.info("Training completed successfully!")
    accelerator.end_training()


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Train SCAN for conditional image generation"
    )
    
    # Model configuration
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to model configuration file")
    parser.add_argument("--exp-name", type=str, default='test', help="Experiment name for logging")
    parser.add_argument("--gpt-ckpt", type=str, default='', help="Path to GPT checkpoint for resuming training")
    parser.add_argument("--ema", action="store_true", help="Use exponential moving average for training")
    parser.add_argument("--no-compile", action="store_true", help="Disable model compilation")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR, help="Directory for saving results")
    
    # Model parameters
    parser.add_argument("--image-size", type=int, choices=[128, 256, 384, 448, 512], default=512, help="Input image size")
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16, help="VQGAN downsampling factor")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of classes for conditional generation")
    parser.add_argument("--num-inference-steps", type=int, default=160, help="Number of inference steps for generation")
    
    # Training parameters
    parser.add_argument("--global-batch-size", type=int, default=16, help="Global batch size across all GPUs")
    parser.add_argument("--max-iters", type=int, default=100000, help="Maximum training iterations")
    parser.add_argument("--global-seed", type=int, default=0, help="Global random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of data loading workers")
    
    # Optimization parameters
    parser.add_argument("--random-ratio", type=float, default=0.25, help="Fixed Random masking ratio, if min and max ratios are set, this is depressed.")
    parser.add_argument("--random-ratio-max", type=float, default=0.1, help="Maximum random masking ratio")
    parser.add_argument("--random-ratio-min", type=float, default=0.5, help="Minimum random masking ratio")
    parser.add_argument("--revise-iter-num", type=int, default=20, help="Number of revision iterations")
    parser.add_argument("--token-num-each-revise", type=int, default=4, help="Number of tokens to revise per iteration")
    
    # Checkpoint and logging
    parser.add_argument("--resume", type=str, default='model.safetensors', help="Path to resume training from checkpoint")
    parser.add_argument("--log-every", type=int, default=1, help="Logging frequency in steps")
    parser.add_argument("--ckpt-every", type=int, default=5000, help="Checkpoint saving frequency in steps")
    parser.add_argument("--keep-last-k", type=int, default=2, help="Number of recent checkpoints to keep")
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["none", "fp16", "bf16"], help="Mixed precision training mode")
    parser.add_argument("--visualize-every", type=int, default=5000, help="Visualization frequency in steps")
    parser.add_argument("--visualize-num", type=int, default=8, help="Number of samples to visualize")
    
    # Model checkpoints
    parser.add_argument("--vq-ckpt", type=str, default=DEFAULT_VQ_CKPT, help="Path to VQGAN checkpoint")
    
    # Data parameters
    parser.add_argument("--dataset", type=str, choices=['imagenetraw', 't2i', 'video'], default='imagenetraw', help="Dataset type")
    parser.add_argument("--data-path", type=str, help="Path to training data")
    parser.add_argument("--data-anno", type=str, help="Path to data annotations")
    parser.add_argument("--imagenet-class", type=str, help="Path to ImageNet class labels")
    
    # Additional data parameters
    parser.add_argument('--nframes', default=1, type=int, help="Number of frames for video data, set 1 for image")
    parser.add_argument('--return-idx', type=bool, default=False, help="Return indices")
    parser.add_argument('--return-class-text', type=bool, default=False, help="Return class text")
    
    # Distributed training
    parser.add_argument("--parallel", action="store_true", default=False, help="Enable distributed training")
    
    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    main(args)