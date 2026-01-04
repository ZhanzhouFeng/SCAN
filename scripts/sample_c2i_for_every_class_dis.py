#!/usr/bin/env python3
"""
Class-conditional image generation script for SCAN model.

Modified from:
    - LLaMAGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/sample/sample_c2i_ddp.py
    - DiT: https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py

This script generates images for each class using the SCAN autoregressive model
with self-correction mechanism and parallel decoding.
"""

import os
import sys
import time
import math
import argparse
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.distributed as dist
from omegaconf import OmegaConf

# Add project root to path
sys.path.append("./")

from SCAN.dataset.builder import build_dataset
from SCAN.utils.distributed import init_distributed_mode, is_main_process
from SCAN.util import instantiate_from_config, load_safetensors
from SCAN.model.utils import calculate_num_query_tokens_for_parallel_decoding_my


def is_directory_empty(directory: str) -> bool:
    """Check if a directory is empty."""
    return len(os.listdir(directory)) == 0


def create_npz_from_samples(sample_dir: str, num_samples: int = 50_000) -> str:
    """
    Build a single .npz file from a folder of .png samples.
    
    Args:
        sample_dir: Directory containing PNG samples
        num_samples: Number of samples to include
    
    Returns:
        Path to the created .npz file
    """
    samples = []
    for i in tqdm(range(num_samples), desc="Building .npz file"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    
    samples = np.stack(samples)
    assert samples.shape == (num_samples, samples.shape[1], samples.shape[2], 3)
    
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


class ModelInitializer:
    """Handles model initialization and configuration."""
    
    def __init__(self, args, device: torch.device):
        self.args = args
        self.device = device
        self.config = OmegaConf.load(args.config)
        
    def initialize_tokenizer(self) -> torch.nn.Module:
        """Initialize and load tokenizer model."""
        tokenizer = instantiate_from_config(self.config.tokenizer).to(self.device).eval()
        
        if self.args.vq_ckpt is not None:
            ckpt = torch.load(self.args.vq_ckpt, map_location="cpu")
            state_dict = ckpt['model'] if 'model' in ckpt else ckpt
            tokenizer.load_state_dict(state_dict)
            
        return tokenizer
    
    def initialize_gpt_model(self) -> torch.nn.Module:
        """Initialize and load GPT model."""
        precision_map = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
        precision = precision_map[self.args.precision]
        
        gpt_model = instantiate_from_config(self.config.ar_model).to(
            device=self.device, dtype=precision
        )
        
        if self.args.gpt_ckpt is not None:
            model_weight = load_safetensors(self.args.gpt_ckpt)
            msg = gpt_model.load_state_dict(model_weight, strict=True)
            print(f"Loaded GPT model: {msg}")
        else:
            print("Using randomly initialized GPT model")
            
        gpt_model.eval()
        return gpt_model


class AttentionMaskBuilder:
    """Builds attention masks for generation."""
    
    @staticmethod
    def build_mask(
        image_size: int, 
        downsample_size: int, 
        num_inference_steps: int, 
        device: torch.device
    ) -> torch.Tensor:
        """Build attention mask for generation."""
        token_num_per_iter = calculate_num_query_tokens_for_parallel_decoding_my(
            num_inference_steps, (image_size // downsample_size) ** 2
        )
        
        num_cond_tokens = 1
        max_seq_length = (image_size // downsample_size) ** 2 * 2 + num_cond_tokens
        
        attn_mask = torch.zeros(max_seq_length, max_seq_length).to(device)
        
        # Build token list: [cond_tokens, interleaved tokens...]
        token_list = [num_cond_tokens]
        token_list += [element for element in token_num_per_iter for _ in range(2)]
        
        # Create causal mask
        mask_pointer = 0
        for token_count in token_list:
            attn_mask[mask_pointer:, mask_pointer:mask_pointer + token_count] = 1
            mask_pointer += token_count
            
        return attn_mask.unsqueeze(0).unsqueeze(0).to(torch.bool)


class SampleGenerator:
    """Handles image generation for each class."""
    
    def __init__(self, args, tokenizer, gpt_model, device):
        self.args = args
        self.tokenizer = tokenizer
        self.gpt_model = gpt_model
        self.device = device
        
    def setup_generation_params(self) -> Tuple[List[float], List[int], List[int]]:
        """Setup generation parameters."""
        cfg_scales = [float(scale) for scale in self.args.cfg_scales.split(",")]
        
        token_num_per_iter = calculate_num_query_tokens_for_parallel_decoding_my(
            self.args.num_inference_steps, (self.args.image_size // self.args.downsample_size) ** 2
        )
        
        revise_token_nums = [self.args.token_num_each_revise] * self.args.revise_iter_num
        
        return cfg_scales, token_num_per_iter, revise_token_nums
    
    def generate_class_samples(self, class_idx: int, output_dir: str) -> None:
        """Generate samples for a specific class."""
        if os.path.exists(output_dir) and not is_directory_empty(output_dir):
            print(f"Skipping class {class_idx}: samples already exist")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        start_time = time.time()
        
        # Create class indices
        batch_size = self.args.per_proc_batch_size
        class_indices = torch.ones(batch_size, device=self.device, dtype=torch.int64) * class_idx
        
        # Setup generation parameters
        cfg_scales, token_num_per_iter, revise_token_nums = self.setup_generation_params()
        
        # Build attention mask
        attn_mask = AttentionMaskBuilder.build_mask(
            self.args.image_size, 
            self.args.downsample_size, 
            self.args.num_inference_steps, 
            self.device
        )
        
        # Generate samples
        indices, _ = self.gpt_model.generate_embs_then_tokens(
            cond=class_indices,
            token_order=None,
            cfg_scales=tuple(cfg_scales),
            num_inference_steps=self.args.num_inference_steps,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            token_num_per_iter=token_num_per_iter,
            mask=attn_mask,
            revise_token_nums=revise_token_nums,
        )
        
        # Decode to images
        samples = self.tokenizer.decode_codes_to_img(indices, self.args.image_size_eval)
        
        print(f"Generated {samples.shape[0]} samples for class {class_idx} "
              f"in {time.time()-start_time:.2f} seconds")
        
        # Save samples
        for i, sample in enumerate(samples):
            index = i + class_idx * self.args.per_proc_batch_size
            Image.fromarray(sample).save(f"{output_dir}/{index:06d}.png")


def setup_distributed_mode(args) -> Tuple[int, torch.device]:
    """Setup distributed training mode."""
    if args.parallel:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        
        seed = args.global_seed * dist.get_world_size() + rank
        torch.manual_seed(seed)
        torch.cuda.set_device(device)
        
        print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}")
    else:
        rank = 0
        device = torch.device("cuda")
        torch.manual_seed(args.global_seed)
        print(f"Starting rank={rank}, seed={args.global_seed}, world_size=1")
    
    return rank, device


def create_output_directory(args, rank: int) -> str:
    """Create output directory for samples."""
    ckpt_name = (
        os.path.basename(args.gpt_ckpt)
        .replace(".pth", "").replace(".pt", "").replace(".safetensors", "")
    )
    
    folder_name = (
        f"{ckpt_name}-size-{args.image_size}-size-{args.image_size_eval}-"
        f"cfg-{args.cfg_scales}-seed-{args.global_seed}"
    )
    folder_name = os.path.join(args.exp_name, folder_name)
    sample_dir = os.path.join(args.sample_dir, folder_name)
    
    if rank == 0:
        os.makedirs(sample_dir, exist_ok=True)
        print(f"Saving samples to: {sample_dir}")
    
    if args.parallel:
        dist.barrier()
    
    return sample_dir


def main(args):
    """Main execution function."""
    # Validate CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for sampling")
    
    torch.set_grad_enabled(False)
    
    # Setup distributed mode
    rank, device = setup_distributed_mode(args)
    
    # Initialize models
    initializer = ModelInitializer(args, device)
    tokenizer = initializer.initialize_tokenizer()
    gpt_model = initializer.initialize_gpt_model()
    
    # Create output directory
    sample_dir = create_output_directory(args, rank)
    
    # Generate samples for each class
    generator = SampleGenerator(args, tokenizer, gpt_model, device)
    
    progress_bar = tqdm(range(args.num_classes), disable=rank != 0)
    
    for class_idx in progress_bar:
        if progress_bar is not None:
            progress_bar.set_description(f"Generating class {class_idx}")
        
        class_output_dir = os.path.join(sample_dir, str(class_idx))
        generator.generate_class_samples(class_idx, class_output_dir)
        
        if args.debug:
            break  # Stop after first class in debug mode


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate images for each class using SCAN model")
    
    # Model configuration
    parser.add_argument("--config", type=str, 
                       default="configs/SCAN/SCANmaskar_xl_0.7b_llamagen_dis.yaml",
                       help="Model configuration file")
    parser.add_argument("--gpt-ckpt", type=str, default='model.safetensors',
                       help="Path to GPT model checkpoint")
    parser.add_argument("--vq-ckpt", type=str, 
                       default='/new_shanghai/fengzheng.fzz/models/LlamaGen/vq_ds16_c2i.pt',
                       help="Path to VQ tokenizer checkpoint")
    
    # Generation parameters
    parser.add_argument("--image-size", type=int, choices=[128, 256, 384, 512], default=256,
                       help="Image size for generation")
    parser.add_argument("--image-size-eval", type=int, choices=[128, 256, 384, 512], default=256,
                       help="Image size for evaluation")
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16,
                       help="Downsample factor")
    parser.add_argument("--num-classes", type=int, default=1000,
                       help="Number of classes")
    parser.add_argument("--cfg-scales", type=str, default="1.0,4.0",
                       help="Classifier-free guidance scales")
    
    # Sampling parameters
    parser.add_argument("--num-inference-steps", type=int, default=88,
                       help="Number of inference steps")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=0,
                       help="Top-k sampling parameter")
    parser.add_argument("--top-p", type=float, default=1.0,
                       help="Top-p sampling parameter")
    
    # Revision parameters
    parser.add_argument("--revise-iter-num", type=int, default=10,
                       help="Number of revision iterations")
    parser.add_argument("--token-num-each-revise", type=int, default=4,
                       help="Number of tokens to revise per iteration")
    
    # Batch and output
    parser.add_argument("--per-proc-batch-size", type=int, default=64,
                       help="Batch size per process")
    parser.add_argument("--sample-dir", type=str, default="./generated_imgs",
                       help="Directory to save samples")
    parser.add_argument("--exp-name", type=str, default='test_noParallel_zscan',
                       help="Experiment name")
    parser.add_argument("--num-fid-samples", type=int, default=100,
                       help="Number of FID samples")
    
    # System parameters
    parser.add_argument("--precision", type=str, default="bf16",
                       choices=["none", "fp16", "bf16"], help="Model precision")
    parser.add_argument("--global-seed", type=int, default=0,
                       help="Global random seed")
    parser.add_argument("--parallel", action="store_true", default=False,
                       help="Use distributed training")
    parser.add_argument("--debug", action="store_true", default=False,
                       help="Enable debug mode")
    parser.add_argument("--compile", action="store_true", default=True,
                       help="Compile model for faster inference")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)