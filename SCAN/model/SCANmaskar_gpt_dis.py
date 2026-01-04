#!/usr/bin/env python3
"""
SCAN GPT Model for conditional image generation with self-correction mechanism.

Modified from:
    - LLaMAGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/models/gpt.py
    - VQGAN: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
    - DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
    - nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
    - llama: https://github.com/facebookresearch/llama/blob/main/llama/model.py
    - gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
    - PixArt: https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
"""

import math
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from .generate import sample
from .llamagen_gpt import (
    CaptionEmbedder,
    FeedForward,
    KVCache,
    LabelEmbedder,
    MLP,
    RMSNorm,
    apply_rotary_emb,
    find_multiple,
    precompute_freqs_cis_2d,
)
from .utils import (
    DropPath,
    calculate_num_query_tokens_for_parallel_decoding,
    embs_then_tokens,
)


def batch_apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to input tensor in batch mode.
    
    Args:
        x: Input tensor of shape (bs, seq_len, n_head, head_dim)
        freqs_cis: Frequency tensor of shape (bs, seq_len, head_dim // 2, 2)
        
    Returns:
        Tensor with rotary embedding applied
    """
    bs, seq_len, n_head, head_dim = x.shape
    xshaped = x.float().reshape(*x.shape[:-1], head_dim // 2, 2)
    freqs_cis = freqs_cis.view(bs, xshaped.size(1), 1, xshaped.size(3), 2)
    
    x_out = torch.stack([
        xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
        xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    
    return x_out.flatten(3).type_as(x)


def random_permutation(x: torch.Tensor) -> torch.Tensor:
    """Apply random permutation to input tensor along sequence dimension."""
    indices = torch.randperm(x.size(1))
    return x[:, indices]


def generate_mask_tensor(batch_size: int, sequence_length: int, mask_ratio: float) -> torch.Tensor:
    """Generate random mask tensor for token masking.
    
    Args:
        batch_size: Batch size
        sequence_length: Length of sequence
        mask_ratio: Ratio of tokens to mask (0.0 to 1.0)
        
    Returns:
        Mask tensor of shape (batch_size, sequence_length)
    """
    ones_count = math.ceil((1 - mask_ratio) * batch_size * sequence_length)
    mask_values = torch.cat([
        torch.ones(ones_count),
        torch.zeros(batch_size * sequence_length - ones_count)
    ])
    
    # Shuffle for randomness
    mask_values = mask_values[torch.randperm(mask_values.size(0))]
    return mask_values.view(batch_size, sequence_length)


class Attention(nn.Module):
    """Attention module with KV cache support for efficient inference."""
    
    def __init__(
        self,
        dim: int,
        n_head: int,
        n_kv_head: int,
        attn_dropout_p: float,
        resid_dropout_p: float,
    ):
        super().__init__()
        assert dim % n_head == 0, f"dim {dim} must be divisible by n_head {n_head}"
        
        self.dim = dim
        self.head_dim = dim // n_head
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # Key, query, value projections
        self.wqkv = nn.Linear(dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.kv_cache = None

        # Regularization
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor = None,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of attention module.
        
        Args:
            x: Input tensor [bsz, seqlen, dim]
            freqs_cis: Frequency embeddings [bsz, seqlen, head_dim // 2, 2]
            input_pos: Position indices for KV cache [seqlen]
            mask: Attention mask [bsz, seqlen, seqlen]
            
        Returns:
            Output tensor [bsz, seqlen, dim]
        """
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        
        # Split into query, key, value
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        # Reshape for multi-head attention
        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)

        # Apply rotary embedding
        xq = batch_apply_rotary_emb(xq, freqs_cis)
        xk = batch_apply_rotary_emb(xk, freqs_cis)

        # Transpose for attention computation
        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        # Handle KV cache
        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
            max_pos = torch.max(input_pos) + 1
            keys = keys[:, :, :max_pos]
            values = values[:, :, :max_pos]
            if mask is not None:
                mask = mask[:, :, :, :max_pos]
        else:
            keys, values = xk, xv

        # Repeat KV heads if necessary
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        # Compute attention
        output = F.scaled_dot_product_attention(
            xq,
            keys,
            values,
            attn_mask=mask,
            is_causal=(mask is None),
            dropout_p=self.attn_dropout_p if self.training else 0,
        )

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        return self.resid_dropout(self.wo(output))


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward layers."""
    
    def __init__(
        self,
        dim: int = 4096,
        n_layer: int = 32,
        n_head: int = 32,
        n_kv_head: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        rope_base: int = 10000,
        norm_eps: float = 1e-5,
        token_dropout_p: float = 0.1,
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.1,
        ffn_dropout_p: float = 0.1,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.attention = Attention(
            dim, n_head, n_kv_head, attn_dropout_p, resid_dropout_p
        )
        self.feed_forward = FeedForward(
            dim, ffn_dim_multiplier, multiple_of, ffn_dropout_p
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of transformer block."""
        h = x + self.drop_path(
            self.attention(self.attention_norm(x), freqs_cis, start_pos, mask)
        )
        return h + self.drop_path(self.feed_forward(self.ffn_norm(h)))


class SCANARTransformer(nn.Module):
    """SCAN Autoregressive Transformer for conditional image generation."""
    
    def __init__(
        self,
        dim: int = 4096,
        n_layer: int = 32,
        n_head: int = 32,
        n_kv_head: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        rope_base: int = 10000,
        norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        token_dropout_p: float = 0.1,
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.1,
        ffn_dropout_p: float = 0.1,
        drop_path_rate: float = 0.0,
        num_classes: int = 1000,
        caption_dim: int = 2048,
        class_dropout_prob: float = 0.1,
        model_type: str = "c2i",
        vocab_size: int = 16384,
        cls_token_num: int = 1,
        block_size: int = 256,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
        position_order: str = "random",
        num_inference_steps: int = 88,
        zero_class_qk: bool = True,
        grad_checkpointing: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.rope_base = rope_base
        self.norm_eps = norm_eps
        self.token_dropout_p = token_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.ffn_dropout_p = ffn_dropout_p
        self.drop_path_rate = drop_path_rate
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_classes = num_classes
        self.model_type = model_type
        self.cls_token_num = cls_token_num
        self.position_order = position_order
        self.num_inference_steps = num_inference_steps
        self.zero_class_qk = zero_class_qk
        self.grad_checkpointing = grad_checkpointing
        self.initializer_range = initializer_range

        # Initialize embeddings based on model type
        if self.model_type == "c2i":
            self.cls_embedding = LabelEmbedder(num_classes, dim, class_dropout_prob)
        elif self.model_type == "t2i":
            self.cls_embedding = CaptionEmbedder(caption_dim, dim, class_dropout_prob)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.tok_dropout = nn.Dropout(token_dropout_p)

        # Initialize transformer layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layer)]
        self.layers = nn.ModuleList()
        for layer_id in range(n_layer):
            self.layers.append(TransformerBlock(
                dim=dim,
                n_layer=n_layer,
                n_head=n_head,
                n_kv_head=n_kv_head,
                multiple_of=multiple_of,
                ffn_dim_multiplier=ffn_dim_multiplier,
                rope_base=rope_base,
                norm_eps=norm_eps,
                token_dropout_p=token_dropout_p,
                attn_dropout_p=attn_dropout_p,
                resid_dropout_p=resid_dropout_p,
                ffn_dropout_p=ffn_dropout_p,
                drop_path=dpr[layer_id],
            ))

        # Output layers
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        # Discriminator layers
        self.dis_norm = RMSNorm(dim, eps=norm_eps)
        self.dis_output = nn.Linear(dim, 2, bias=False)

        # 2D rotary position embedding
        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size, "block_size must be a perfect square"
        self.freqs_cis = precompute_freqs_cis_2d(
            grid_size, self.dim // self.n_head, self.rope_base, self.cls_token_num
        )

        # KV cache initialization
        self.max_batch_size = -1
        self.max_seq_length = -1

        # SCAN-specific parameters
        self.pos_instruct_embeddings = nn.Parameter(
            torch.randn(1, self.dim) * self.initializer_range
        )

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights."""
        self.apply(self._init_weights)
        
        # Zero-out output layers
        nn.init.constant_(self.output.weight, 0)
        nn.init.constant_(self.dis_output.weight, 0)

    def _init_weights(self, module):
        """Initialize weights for individual modules."""
        std = self.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def setup_caches(self, max_batch_size: int, max_seq_length: int, dtype: torch.dtype, mask: Optional[torch.Tensor] = None):
        """Setup KV caches for inference."""
        head_dim = self.dim // self.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        
        for layer in self.layers:
            layer.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.n_head, head_dim, dtype
            )

        if mask is None:
            causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
            self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        else:
            self.causal_mask = mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)

        # Recompute freqs_cis for new sequence length
        grid_size = int(self.block_size ** 0.5)
        self.freqs_cis = precompute_freqs_cis_2d(
            grid_size, self.dim // self.n_head, self.rope_base, self.cls_token_num
        )

    def remove_caches(self):
        """Remove KV caches to free memory."""
        for layer in self.layers:
            layer.attention.kv_cache = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def forward(
        self,
        idx: torch.Tensor,
        cond_idx: torch.Tensor,
        token_order: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        token_num_per_iter: Optional[List[int]] = None,
        discriminator_target: Optional[torch.Tensor] = None,
        revise_iter_num: Optional[int] = None,
    ):
        """Main forward pass."""
        if idx is not None and cond_idx is not None:
            return self.forward_train(
                idx, cond_idx, token_order, input_pos, targets, mask, valid,
                token_num_per_iter=token_num_per_iter,
                discriminator_target=discriminator_target,
                revise_iter_num=revise_iter_num
            )
        else:
            raise ValueError("idx and cond_idx cannot be both None")

    def forward_train(
        self,
        idx: torch.Tensor,
        cond_idx: torch.Tensor,
        token_order: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        token_num_per_iter: Optional[List[int]] = None,
        discriminator_target: Optional[torch.Tensor] = None,
        revise_iter_num: Optional[int] = None,
    ):
        """Training forward pass.
        
        Args:
            idx: GT image tokens [bsz, seq_len]
            cond_idx: Conditional tokens [bsz, cls_token_num]
            token_order: Position order [bsz, seq_len]
            targets: Target tokens [bsz, seq_len]
            discriminator_target: Discriminator targets [bsz, seq_len]
        """
        batch_size = idx.shape[0]
        
        # Prepare token order
        if token_order is None:
            token_order = self._prepare_token_order(batch_size, idx.device)
        
        # Permute tokens according to order
        idx = self._permute_tokens(idx, token_order)
        targets = self._permute_tokens(targets, token_order)
        discriminator_target = self._permute_tokens(discriminator_target, token_order)

        # Prepare revised tokens
        revised_num_sum = sum(token_num_per_iter[-revise_iter_num:])
        append_idx = self._prepare_revision_indices(discriminator_target, revised_num_sum)
        
        revised_target = self._permute_tokens(targets, append_idx)
        revised_mask = generate_mask_tensor(batch_size, revised_num_sum, 0.2).to(targets.device)
        revised_token_order = self._permute_tokens(token_order, append_idx)
        
        # Generate revised tokens
        random_tensor = torch.randint(
            low=1, high=self.vocab_size, size=(batch_size, revised_num_sum), device=targets.device
        )
        revised_patch_input = revised_target * revised_mask + random_tensor * (1 - revised_mask)

        # Concatenate tokens
        idx = torch.cat([idx, revised_patch_input.to(idx.dtype)], dim=1)
        targets = torch.cat([targets, revised_target], dim=1)
        discriminator_target = torch.cat([
            discriminator_target, revised_mask.to(discriminator_target.dtype)
        ], dim=1)
        token_order = torch.cat([token_order, revised_token_order], dim=1)

        # Prepare embeddings
        self.freqs_cis = self.freqs_cis.to(cond_idx.device)
        cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:, :self.cls_token_num]
        
        token_embeddings = self.tok_embeddings(idx)
        token_embeddings = self.tok_dropout(token_embeddings)
        position_instruction_tokens = self.get_position_instruction_tokens(token_order)

        # Combine embeddings
        h = torch.cat([
            cond_embeddings,
            embs_then_tokens(position_instruction_tokens, token_embeddings, token_num_per_iter)
        ], dim=1)

        # Prepare frequency embeddings
        token_freqs_cis = self.freqs_cis[self.cls_token_num:].clone().to(token_order.device)[token_order]
        freqs_cis = torch.cat([
            self.freqs_cis[:self.cls_token_num].unsqueeze(0).repeat(batch_size, 1, 1, 1),
            embs_then_tokens(token_freqs_cis, token_freqs_cis, token_num_per_iter)
        ], dim=1)

        # Forward through transformer layers
        for layer in self.layers:
            if self.grad_checkpointing:
                h = checkpoint(layer, h, freqs_cis, input_pos, mask, use_reentrant=False)
            else:
                h = layer(h, freqs_cis, input_pos, mask)

        # Extract embeddings and discriminator outputs
        h = h[:, self.cls_token_num:]
        embs, dis = self._extract_embeddings(h, token_num_per_iter)
        h = embs.contiguous()

        # Compute outputs
        h = self.norm(h)
        logits = self.output(h).float()
        token_logits = logits

        # Compute losses
        loss = self._compute_main_loss(token_logits, targets, valid)
        dis_loss = self._compute_discriminator_loss(dis, discriminator_target)

        return token_logits, loss, token_order, dis_loss

    def _prepare_token_order(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Prepare token order based on position_order setting."""
        token_order = torch.arange(self.block_size, device=device, dtype=torch.long)
        token_order = token_order.unsqueeze(0).repeat(batch_size, 1)
        
        if self.position_order == "random":
            for i in range(batch_size):
                token_order[i] = token_order[i][torch.randperm(self.block_size)]
        elif self.position_order == "raster":
            pass  # Keep raster order
        else:
            raise ValueError(f"Invalid position order: {self.position_order}")
        
        return token_order.contiguous()

    def _permute_tokens(self, tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Permute tokens according to given indices."""
        return torch.gather(tokens.unsqueeze(-1), 1, indices.unsqueeze(-1)).squeeze(-1)

    def _prepare_revision_indices(self, discriminator_target: torch.Tensor, revised_num_sum: int) -> torch.Tensor:
        """Prepare indices for revision tokens."""
        _, indices = torch.sort(discriminator_target)
        indices = indices[:, :revised_num_sum]
        return random_permutation(indices)

    def _extract_embeddings(self, h: torch.Tensor, token_num_per_iter: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract embeddings and discriminator outputs."""
        embs, dis = [], []
        pointer = 0
        
        for token_count in token_num_per_iter:
            embs.append(h[:, pointer:pointer + token_count])
            dis.append(h[:, pointer + token_count:pointer + token_count * 2])
            pointer += token_count * 2
        
        return torch.cat(embs, dim=1), torch.cat(dis, dim=1).contiguous()

    def _compute_main_loss(self, logits: torch.Tensor, targets: torch.Tensor, valid: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute main training loss."""
        if valid is not None:
            loss_all = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
            )
            valid_all = valid[:, None].repeat(1, targets.shape[1]).view(-1)
            return (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        else:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    def _compute_discriminator_loss(self, dis: torch.Tensor, discriminator_target: torch.Tensor) -> torch.Tensor:
        """Compute discriminator loss."""
        if discriminator_target is not None:
            dis = self.dis_norm(dis)
            dis_logits = self.dis_output(dis).float()
            return F.cross_entropy(dis_logits.view(-1, dis_logits.size(-1)), discriminator_target.view(-1))
        return torch.tensor(0.0, device=dis.device)

    def forward_inference(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        input_pos: torch.Tensor,
        condi_length: int = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inference forward pass.
        
        Args:
            x: Input tokens [bs, query_num, dim]
            freqs_cis: Frequency embeddings [bs, query_num, n_head, dim // n_head]
            input_pos: Position indices [query_num]
        """
        bs = x.shape[0]
        if mask is None:
            mask = self.causal_mask[:bs, None, input_pos]
        
        h = x
        for layer in self.layers:
            h = layer(h, freqs_cis, start_pos=input_pos, mask=mask)
        
        if condi_length is not None:
            next_tokens = self.norm(h[:, :])
            logits = self.output(next_tokens).float()
            
            eval_tokens = self.dis_norm(h[:, :condi_length])
            eval_tokens = self.dis_output(eval_tokens).float()
            return logits, eval_tokens
        else:
            h = self.norm(h)
            return self.output(h).float()

    def get_position_instruction_tokens(self, token_order: torch.Tensor) -> torch.Tensor:
        """Generate position instruction tokens."""
        position_instruct_tokens = self.pos_instruct_embeddings.view(
            1, 1, self.n_head, self.dim // self.n_head
        )
        position_instruct_tokens = position_instruct_tokens.repeat(
            token_order.shape[0], token_order.shape[1], 1, 1
        )
        
        # Apply rotary embedding
        position_instruct_freqs_cis = self.freqs_cis[self.cls_token_num:].clone().to(token_order.device)[token_order]
        position_instruct_tokens = batch_apply_rotary_emb(position_instruct_tokens, position_instruct_freqs_cis)
        
        return position_instruct_tokens.view(
            token_order.shape[0], token_order.shape[1], self.dim
        ).contiguous()

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        """Get list of modules for FSDP wrapping."""
        return list(self.layers)

    def configure_optimizer(
        self, lr: float, weight_decay: float, beta1: float, beta2: float, max_grad_norm: float, **kwargs
    ) -> torch.optim.Optimizer:
        """Configure AdamW optimizer with parameter grouping."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Group parameters by dimension for weight decay
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        
        import inspect
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra_args = dict(fused=True) if fused_available else dict()
        
        return torch.optim.AdamW(
            optim_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            **extra_args
        )

    def generate(
        self,
        cond: torch.Tensor,
        token_order: torch.Tensor = None,
        cfg_scales: Tuple[float, float] = (1.0, 1.0),
        num_inference_steps: int = 88,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Generate image tokens using parallel decoding.
        
        Args:
            cond: Conditional tokens [batch_size, seq_len]
            token_order: Position order for each token [batch_size, seq_len]
            cfg_scales: (cfg_scale_start, cfg_scale_end) for classifier-free guidance
            num_inference_steps: Number of inference steps
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            
        Returns:
            Generated token indices [batch_size, block_size]
        """
        batch_size = cond.shape[0]
        
        # Step 1: Prepare token order
        token_order = self._prepare_inference_token_order(batch_size, cond.device, token_order)
        
        # Step 2: Prepare embeddings and frequency tensors
        position_tokens = self.get_position_instruction_tokens(token_order)
        freq_embeddings = self.freqs_cis[self.cls_token_num:].clone().to(token_order.device)[token_order]
        
        # Step 3: Setup classifier-free guidance
        batch_size, cond_tokens, position_tokens, freq_embeddings = self._setup_cfg(
            cond, position_tokens, freq_embeddings, cfg_scales
        )
        
        # Step 4: Setup KV cache for inference
        max_seq_len = cond_tokens.shape[1] + self.block_size * 2
        self.setup_caches(
            max_batch_size=batch_size,
            max_seq_length=max_seq_len,
            dtype=self.tok_embeddings.weight.dtype,
            mask=None
        )
        
        # Step 5: Generate tokens with parallel decoding
        return self._parallel_generation(
            cond_tokens, position_tokens, freq_embeddings,
            cfg_scales, num_inference_steps, temperature, top_k, top_p
        )

    def generate_embs_then_tokens(
        self,
        cond: torch.Tensor,
        token_order: torch.Tensor = None,
        cfg_scales: Tuple[float, float] = (1.0, 1.0),
        num_inference_steps: int = 88,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        token_num_per_iter: List[int] = None,
        mask: torch.Tensor = None,
        revise_token_nums: List[int] = None,
        output_intermediate_results: bool = False,
    ) -> torch.Tensor:
        """Generate tokens with embeddings and self-correction mechanism.
        
        Args:
            cond: Conditional tokens [batch_size, seq_len]
            token_order: Position order for each token [batch_size, seq_len]
            cfg_scales: (cfg_scale_start, cfg_scale_end) for classifier-free guidance
            num_inference_steps: Number of inference steps
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            token_num_per_iter: Number of tokens to generate per iteration
            mask: Attention mask
            revise_token_nums: Number of tokens to revise in each correction step
            output_intermediate_results: Whether to return intermediate results
            
        Returns:
            Generated token indices and optionally intermediate results
        """
        batch_size = cond.shape[0]
        
        # Step 1: Prepare token order
        token_order = self._prepare_inference_token_order(batch_size, cond.device, token_order)
        
        # Step 2: Prepare embeddings and frequency tensors
        position_tokens = self.get_position_instruction_tokens(token_order)
        freq_embeddings = self.freqs_cis[self.cls_token_num:].clone().to(token_order.device)[token_order]
        
        # Step 3: Setup classifier-free guidance
        batch_size, cond_tokens, position_tokens, freq_embeddings = self._setup_cfg(
            cond, position_tokens, freq_embeddings, cfg_scales
        )
        
        # Step 4: Setup KV cache for inference
        max_seq_len = cond_tokens.shape[1] + self.block_size * 2 + sum(revise_token_nums or [0]) * 2
        self.setup_caches(
            max_batch_size=batch_size,
            max_seq_length=max_seq_len,
            dtype=self.tok_embeddings.weight.dtype,
            mask=mask.squeeze() if mask is not None else None
        )
        
        # Step 5: Generate tokens with parallel decoding and self-correction
        return self._generate_with_revision(
            cond_tokens, position_tokens, freq_embeddings,
            cfg_scales, temperature, top_k, top_p,
            token_num_per_iter, revise_token_nums, output_intermediate_results
        )

    def _prepare_inference_token_order(
        self, batch_size: int, device: torch.device, token_order: torch.Tensor = None
    ) -> torch.Tensor:
        """Prepare token order for inference."""
        if token_order is not None:
            assert token_order.shape == (batch_size, self.block_size)
            return token_order
        
        # Create default token order
        token_order = torch.arange(self.block_size, device=device)
        token_order = token_order.unsqueeze(0).repeat(batch_size, 1)
        
        # Apply random permutation if needed
        if self.position_order == "random":
            for i in range(batch_size):
                token_order[i] = token_order[i][torch.randperm(self.block_size)]
        
        return token_order.contiguous()

    def _setup_cfg(
        self, cond: torch.Tensor, position_tokens: torch.Tensor, freq_embeddings: torch.Tensor,
        cfg_scales: Tuple[float, float]
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Setup classifier-free guidance."""
        batch_size = cond.shape[0]
        
        if cfg_scales[-1] <= 1.0:
            # No CFG
            cond_tokens = self.cls_embedding(cond, train=False)
            return batch_size, cond_tokens, position_tokens, freq_embeddings
        
        # Apply CFG by concatenating conditional and unconditional inputs
        cond_null = torch.ones_like(cond) * self.num_classes
        cond_combined = torch.cat([cond, cond_null])
        
        # Duplicate position tokens and frequency embeddings
        position_tokens = torch.cat([position_tokens, position_tokens])
        freq_embeddings = torch.cat([freq_embeddings, freq_embeddings])
        
        # Create conditional tokens
        cond_tokens = self.cls_embedding(cond_combined, train=False)
        
        return batch_size * 2, cond_tokens, position_tokens, freq_embeddings

    def _parallel_generation(
        self,
        cond_tokens: torch.Tensor,
        position_tokens: torch.Tensor,
        freq_embeddings: torch.Tensor,
        cfg_scales: Tuple[float, float],
        num_inference_steps: int,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> torch.Tensor:
        """Generate tokens using parallel decoding."""
        batch_size = cond_tokens.shape[0]
        result_indices = torch.zeros((batch_size, self.block_size), dtype=torch.long, device=cond_tokens.device)
        
        # Initialize generation state
        current_step = 0
        tokens_per_step = 1
        start_idx = 0
        token_counts = []
        
        # Prepare initial input
        x = torch.cat([
            cond_tokens,
            position_tokens[:, start_idx:start_idx + tokens_per_step]
        ], dim=1)
        
        freqs_cis = torch.cat([
            self.freqs_cis[:self.cls_token_num].unsqueeze(0).repeat(batch_size, 1, 1, 1),
            freq_embeddings[:, start_idx:start_idx + tokens_per_step]
        ], dim=1)
        
        input_pos = torch.arange(0, x.shape[1], device=cond_tokens.device)
        
        # Generation loop
        while start_idx < self.block_size:
            # Generate tokens for current step
            logits = self.forward_inference(x, freqs_cis, input_pos)
            
            # Apply classifier-free guidance
            if cfg_scales[-1] > 1.0:
                cfg_scale = cfg_scales[0] + (cfg_scales[-1] - cfg_scales[0]) * start_idx / self.block_size
                logits = self._apply_cfg(logits, batch_size, cfg_scale)
            
            # Sample tokens
            step_logits = logits[:, -tokens_per_step:]
            step_indices = self._sample_tokens(step_logits, temperature, top_k, top_p)
            
            # Save generated tokens
            result_indices[:, start_idx:start_idx + tokens_per_step] = step_indices
            
            # Prepare for next step
            next_tokens = self.tok_embeddings(step_indices)
            if cfg_scales[-1] > 1.0:
                next_tokens = torch.cat([next_tokens, next_tokens], dim=0)
            
            current_step += 1
            next_tokens_per_step = calculate_num_query_tokens_for_parallel_decoding(
                current_step, num_inference_steps, self.block_size, start_idx, tokens_per_step
            )
            token_counts.append(next_tokens_per_step)
            
            # Update input for next step
            x, freqs_cis, input_pos = self._prepare_next_step_input(
                next_tokens, position_tokens, freq_embeddings, start_idx, tokens_per_step, next_tokens_per_step
            )
            
            start_idx += tokens_per_step
            tokens_per_step = next_tokens_per_step
            
            if start_idx >= self.block_size:
                break
        
        # Convert back to raster order
        return self._restore_raster_order(result_indices, token_order), token_counts

    def _generate_with_revision(
        self,
        cond_tokens: torch.Tensor,
        position_tokens: torch.Tensor,
        freq_embeddings: torch.Tensor,
        cfg_scales: Tuple[float, float],
        temperature: float,
        top_k: int,
        top_p: float,
        token_num_per_iter: List[int],
        revise_token_nums: List[int],
        output_intermediate_results: bool
    ) -> torch.Tensor:
        """Generate tokens with self-correction mechanism."""
        batch_size = cond_tokens.shape[0]
        
        # Phase 1: Initial generation
        initial_tokens, _ = self._parallel_generation(
            cond_tokens, position_tokens, freq_embeddings,
            cfg_scales, len(token_num_per_iter), temperature, top_k, top_p
        )
        
        # Phase 2: Self-correction
        corrected_tokens = self._apply_self_correction(
            initial_tokens, position_tokens, freq_embeddings,
            cfg_scales, temperature, top_k, top_p, revise_token_nums
        )
        
        # Convert back to raster order
        final_tokens = self._restore_raster_order(corrected_tokens, position_tokens)
        
        return final_tokens, initial_tokens

    def _apply_cfg(self, logits: torch.Tensor, original_batch_size: int, cfg_scale: float) -> torch.Tensor:
        """Apply classifier-free guidance to logits."""
        cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
        return uncond_logits + cfg_scale * (cond_logits - uncond_logits)

    def _sample_tokens(
        self, logits: torch.Tensor, temperature: float, top_k: int, top_p: float
    ) -> torch.Tensor:
        """Sample tokens from logits."""
        batch_size, seq_len, vocab_size = logits.shape
        indices = torch.zeros(batch_size, seq_len, dtype=torch.long, device=logits.device)
        
        for i in range(seq_len):
            indices[:, i:i+1] = sample(
                logits[:, i:i+1], temperature=temperature, top_k=top_k, top_p=top_p
            )[0]
        
        return indices

    def _prepare_next_step_input(
        self, new_tokens: torch.Tensor, position_tokens: torch.Tensor, freq_embeddings: torch.Tensor,
        start_idx: int, prev_step_size: int, next_step_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare input for the next generation step."""
        batch_size = new_tokens.shape[0]
        
        # Create interleaved input: [prev_tokens, position_tokens, new_tokens, next_position_tokens]
        total_length = 2 * prev_step_size - 1 + next_step_size
        
        # Initialize tensors
        x = torch.zeros(batch_size, total_length, self.dim, dtype=new_tokens.dtype, device=new_tokens.device)
        freqs_cis = torch.zeros(
            batch_size, total_length, *self.freqs_cis.shape[-2:], 
            dtype=freq_embeddings.dtype, device=freq_embeddings.device
        )
        
        # Fill in the tensors with proper interleaving
        next_start_idx = start_idx + prev_step_size
        
        # Previous tokens and positions
        x[:, :1] = new_tokens[:, :1]
        x[:, 1:2*prev_step_size-1:2] = position_tokens[:, start_idx+1:start_idx+prev_step_size]
        x[:, 2:2*prev_step_size:2] = new_tokens[:, 1:prev_step_size]
        x[:, 2*prev_step_size-1:] = position_tokens[:, next_start_idx:next_start_idx+next_step_size]
        
        # Frequency embeddings
        freqs_cis[:, :1] = freq_embeddings[:, start_idx:start_idx+1]
        freqs_cis[:, 1:2*prev_step_size-1:2] = freq_embeddings[:, start_idx+1:start_idx+prev_step_size]
        freqs_cis[:, 2:2*prev_step_size:2] = freq_embeddings[:, start_idx+1:start_idx+prev_step_size]
        freqs_cis[:, 2*prev_step_size-1:] = freq_embeddings[:, next_start_idx:next_start_idx+next_step_size]
        
        # Position indices
        last_pos = 2 * start_idx + prev_step_size
        input_pos = torch.arange(total_length, device=new_tokens.device) + last_pos
        
        return x, freqs_cis, input_pos

    def _apply_self_correction(
        self,
        initial_tokens: torch.Tensor,
        position_tokens: torch.Tensor,
        freq_embeddings: torch.Tensor,
        cfg_scales: Tuple[float, float],
        temperature: float,
        top_k: int,
        top_p: float,
        revise_token_nums: List[int]
    ) -> torch.Tensor:
        """Apply self-correction to generated tokens."""
        batch_size = initial_tokens.shape[0]
        corrected_tokens = initial_tokens.clone()
        
        for revise_count in revise_token_nums:
            # Identify tokens to revise based on evaluation scores
            eval_scores = self._evaluate_tokens(corrected_tokens, position_tokens, freq_embeddings)
            _, revise_indices = torch.topk(eval_scores, revise_count, dim=1, largest=False)
            
            # Generate revised tokens
            revised_tokens = self._revise_tokens(
                corrected_tokens, revise_indices, position_tokens, freq_embeddings,
                cfg_scales, temperature, top_k, top_p
            )
            
            # Update tokens based on improvement
            corrected_tokens = self._update_with_revisions(
                corrected_tokens, revised_tokens, revise_indices, eval_scores
            )
        
        return corrected_tokens

    def _evaluate_tokens(
        self, tokens: torch.Tensor, position_tokens: torch.Tensor, freq_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate token quality for revision."""
        # This would contain the actual evaluation logic
        # For now, return dummy scores
        return torch.randn(tokens.shape[0], tokens.shape[1], device=tokens.device)

    def _revise_tokens(
        self, original_tokens: torch.Tensor, revise_indices: torch.Tensor,
        position_tokens: torch.Tensor, freq_embeddings: torch.Tensor,
        cfg_scales: Tuple[float, float], temperature: float, top_k: int, top_p: float
    ) -> torch.Tensor:
        """Generate revised tokens for specified positions."""
        batch_size = original_tokens.shape[0]
        
        # Gather tokens to revise
        selected_positions = torch.gather(
            position_tokens, 1, revise_indices.unsqueeze(-1).expand(-1, -1, position_tokens.size(-1))
        )
        selected_freqs = torch.gather(
            freq_embeddings, 1, revise_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *freq_embeddings.shape[-2:])
        )
        
        # Generate revised tokens
        x = selected_positions
        input_pos = torch.arange(x.shape[1], device=x.device)
        mask = torch.ones(x.shape[1], x.shape[1]).to(x.device)
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).to(torch.bool)
        
        logits = self.forward_inference(x, selected_freqs, input_pos, mask=mask)
        
        # Apply CFG
        if cfg_scales[-1] > 1.0:
            cfg_scale = cfg_scales[-1]
            logits = self._apply_cfg(logits, batch_size, cfg_scale)
        
        return self._sample_tokens(logits, temperature, top_k, top_p)

    def _update_with_revisions(
        self, original_tokens: torch.Tensor, revised_tokens: torch.Tensor,
        revise_indices: torch.Tensor, eval_scores: torch.Tensor
    ) -> torch.Tensor:
        """Update tokens with revised versions based on quality improvement."""
        # This would contain the actual update logic
        # For now, just replace with revised tokens
        updated_tokens = original_tokens.clone()
        for i in range(original_tokens.shape[0]):
            updated_tokens[i, revise_indices[i]] = revised_tokens[i]
        return updated_tokens

    def _restore_raster_order(self, tokens: torch.Tensor, token_order: torch.Tensor) -> torch.Tensor:
        """Restore tokens to raster order from shuffled order."""
        reverse_permutation = torch.argsort(token_order, dim=-1).long()
        return torch.gather(tokens, 1, reverse_permutation)
