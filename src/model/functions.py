# compute_rope_cache and apply_rope are adapted from Sebastian Raschka's
# "Build a Large Language Model From Scratch" (LLMs-from-scratch).
# Source: https://github.com/rasbt/LLMs-from-scratch
# Copyright (c) 2023-2026 Sebastian Raschka.
# Licensed under Apache 2.0; see LICENSE-LLMs-from-scratch.txt.
# Modified by Lintai Hou: configuration-based cache interface,
# variable renaming, and simplifications.

from fractions import Fraction

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def schedule_layers(cfg):
    """
    Evenly interleave sparse and dense layers, prioritizing sparse layers on ties.
    Sparse layers perform no referrals when they are first in the schedule; otherwise, they perform `n_referrals` referrals.
    Input edges are refreshed only after a previous referral, and dense edges are refreshed only after a previous dense layer.
    """
    scheduled = []
    for layer_type, count in enumerate(cfg.n_layers):
        for i in range(count):
            position = Fraction(2 * i + 1, 2 * count)
            scheduled.append((position, layer_type, i))
    scheduled.sort()

    layers = []
    referred = False
    dense_seen = False
    sparse_left = cfg.n_layers[0]
    for l, (_, layer_type, _) in enumerate(scheduled):
        if layer_type == 0:
            info = []
            if l > 0:
                for _ in range(cfg.n_referrals):
                    r_i = cfg.n_input_refreshes if referred else 0
                    r_d = cfg.n_dense_refreshes if dense_seen else 0
                    info.append((r_i, r_d))
                    referred = True
            sparse_left -= 1
        else:
            info = cfg.n_dense_refreshes if sparse_left else 0
            dense_seen = True
        layers.append((layer_type, info))
    return layers


def softplus_0_1(x):
    return F.softplus(x + 0.54132485)


def scale_softmax(x, t):
    logits = t * x.clamp_min(1e-8).log()
    logits = logits.masked_fill(x == 0, -torch.inf)
    return logits.softmax(dim=-1)


def scale_power(x, t):
    scaled = x.clamp_min(1e-8).pow(t)
    scaled = scaled.masked_fill(x == 0, 0.0)
    return scaled / scaled.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _sparsify_triton_warp_configs():
    return [triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)]


@triton.autotune(configs=_sparsify_triton_warp_configs(), key=["SPARSITY", "BLOCK_SIZE"])
@triton.jit
def _sparsify_triton_forward_kernel(
    indices_ptr,
    weights_ptr,
    output_indices_ptr,
    output_weights_ptr,
    output_valid_ptr,
    row_size,
    SPARSITY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    input_base = row * row_size
    output_base = row * SPARSITY

    offsets = tl.arange(0, BLOCK_SIZE)
    valid = offsets < row_size

    indices = tl.load(indices_ptr + input_base + offsets, mask=valid, other=0)

    merged_weights = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    has_earlier_duplicate = tl.zeros([BLOCK_SIZE], dtype=tl.int1)

    for j in tl.range(0, row_size):
        index_j = tl.load(indices_ptr + input_base + j)
        weight_j = tl.load(weights_ptr + input_base + j).to(tl.float32)

        same_index = valid & (indices == index_j)

        merged_weights += tl.where(same_index, weight_j, 0.0)
        has_earlier_duplicate |= same_index & (j < offsets)

    representative = valid & ~has_earlier_duplicate

    scores = tl.where(valid, tl.where(representative, merged_weights, 0.0), float("-inf"))

    for rank in tl.static_range(0, SPARSITY):
        selected_weight, position = tl.max(scores, axis=0, return_indices=True, return_indices_tie_break_left=True)

        selected_is_representative = tl.sum(tl.where(offsets == position, representative.to(tl.int32), 0), axis=0)

        original_index = tl.load(indices_ptr + input_base + position)

        selected_index = tl.where(selected_is_representative != 0, original_index, 0)

        tl.store(output_indices_ptr + output_base + rank, selected_index)
        tl.store(output_weights_ptr + output_base + rank, selected_weight)
        tl.store(output_valid_ptr + output_base + rank, selected_is_representative != 0)

        scores = tl.where(offsets == position, float("-inf"), scores)


@triton.autotune(configs=_sparsify_triton_warp_configs(), key=["SPARSITY", "BLOCK_SIZE"])
@triton.jit
def _sparsify_triton_backward_kernel(
    indices_ptr,
    selected_indices_ptr,
    selected_valid_ptr,
    grad_output_weights_ptr,
    grad_weights_ptr,
    row_size,
    SPARSITY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    input_base = row * row_size
    output_base = row * SPARSITY

    offsets = tl.arange(0, BLOCK_SIZE)
    valid_input = offsets < row_size

    indices = tl.load(indices_ptr + input_base + offsets, mask=valid_input, other=0)

    grad_weights = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for rank in tl.static_range(0, SPARSITY):
        selected_index = tl.load(selected_indices_ptr + output_base + rank)
        selected_valid = tl.load(selected_valid_ptr + output_base + rank) != 0
        selected_grad = tl.load(grad_output_weights_ptr + output_base + rank)

        grad_weights += tl.where(valid_input & selected_valid & (indices == selected_index), selected_grad, 0.0)

    tl.store(grad_weights_ptr + input_base + offsets, grad_weights, mask=valid_input)


class _SparsifyTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, weights, sparsity):
        indices = indices.contiguous()
        weights = weights.contiguous()

        row_size = indices.shape[-1]
        output_shape = indices.shape[:-1] + (sparsity,)
        num_rows = indices.numel() // row_size
        block_size = triton.next_power_of_2(row_size)

        output_indices = torch.empty(output_shape, dtype=torch.int32, device=indices.device)
        output_weights = torch.empty(output_shape, dtype=torch.float32, device=weights.device)
        output_valid = torch.empty(output_shape, dtype=torch.bool, device=indices.device)

        _sparsify_triton_forward_kernel[(num_rows,)](
            indices, weights, output_indices, output_weights, output_valid, row_size, SPARSITY=sparsity, BLOCK_SIZE=block_size
        )

        ctx.save_for_backward(indices, output_indices, output_valid)
        ctx.sparsity = sparsity
        ctx.mark_non_differentiable(output_indices)

        return output_indices, output_weights

    @staticmethod
    def backward(ctx, grad_output_indices, grad_output_weights):
        indices, output_indices, output_valid = ctx.saved_tensors
        grad_output_weights = grad_output_weights.contiguous()

        grad_weights = torch.empty(indices.shape, dtype=grad_output_weights.dtype, device=grad_output_weights.device)

        row_size = indices.shape[-1]
        num_rows = indices.numel() // row_size
        block_size = triton.next_power_of_2(row_size)

        _sparsify_triton_backward_kernel[(num_rows,)](
            indices, output_indices, output_valid, grad_output_weights, grad_weights, row_size, SPARSITY=ctx.sparsity, BLOCK_SIZE=block_size
        )

        return None, grad_weights, None


def sparsify_triton(indices, weights, sparsity):
    sparse_indices, sparse_weights = _SparsifyTriton.apply(indices, weights, sparsity)
    denom = sparse_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    sparse_weights = sparse_weights / denom
    return sparse_indices, sparse_weights


def sparsify_pytorch(indices, weights, sparsity):
    sorted_indices, order = indices.sort(dim=-1)
    sorted_weights = weights.gather(dim=-1, index=order)

    is_new = torch.ones_like(sorted_indices, dtype=torch.bool)
    is_new[..., 1:] = sorted_indices[..., 1:] != sorted_indices[..., :-1]
    segment_ids = is_new.cumsum(dim=-1) - 1

    merged_weights = torch.zeros_like(sorted_weights)
    merged_indices = torch.zeros_like(sorted_indices)
    merged_weights.scatter_add_(dim=-1, index=segment_ids, src=sorted_weights)
    merged_indices.scatter_(dim=-1, index=segment_ids, src=sorted_indices)

    _, top_pos = merged_weights.topk(k=sparsity, dim=-1, sorted=False)

    sparse_weights = merged_weights.gather(dim=-1, index=top_pos)
    sparse_indices = merged_indices.gather(dim=-1, index=top_pos)

    denom = sparse_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    sparse_weights = sparse_weights / denom
    return sparse_indices, sparse_weights


def fetch_select(input, index):
    b, h, n, d = input.shape
    *_, s = index.shape
    input_flat = input.reshape(b * h * n, d)
    bh_offset = torch.arange(b * h, device=index.device, dtype=torch.int32).view(b, h, 1, 1) * n
    flat_index = (index + bh_offset).reshape(-1)
    out = input_flat.index_select(0, flat_index)
    out = out.view(b, h, n, s, d)
    return out


def fetch_index(input, index):
    b, h, _, _ = input.shape
    b_idx = torch.arange(b, device=index.device)[:, None, None, None]
    h_idx = torch.arange(h, device=index.device)[None, :, None, None]
    out = input[b_idx, h_idx, index, :]
    return out


def compute_edge_cache(cfg):
    pos = torch.arange(cfg.seq_length, dtype=torch.int32)
    offsets = torch.arange(cfg.n_edges - 1, -1, -1, dtype=torch.int32)
    indices = torch.zeros((cfg.batch_size, cfg.seq_length, cfg.n_edges, cfg.sparsities[0]), dtype=torch.int32)
    weights = torch.zeros((cfg.batch_size, cfg.seq_length, cfg.n_edges, cfg.sparsities[0]), dtype=torch.float32)
    indices[:, :, :, 0] = (pos[:, None] - offsets[None, :]).clamp_min(0)
    weights[:, :, :, 0] = 1.0
    return indices, weights


def compute_rope_cache(cfg):
    assert cfg.head_size % 2 == 0
    inv_freq = 1.0 / (cfg.rope_base ** (torch.arange(0, cfg.head_size, 2).float() / cfg.head_size))
    positions = torch.arange(cfg.seq_length)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def apply_rope(x, cos, sin):
    batch_size, num_heads, seq_len, head_size = x.shape
    x1 = x[..., : head_size // 2]
    x2 = x[..., head_size // 2 :]
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.to(dtype=x.dtype)
