# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Optional, Tuple

import numpy as np

import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention Is All You Need paper, generalized to work on images.
    """

    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.cache = {}

    def _encode_xy(self, x, y):
        # The positions are expected to be normalized
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2
        ).flatten(1)
        pos_y = torch.stack(
            (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2
        ).flatten(1)
        return pos_x, pos_y

    @torch.no_grad()
    def encode_boxes(self, x, y, w, h):
        pos_x, pos_y = self._encode_xy(x, y)
        pos = torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)
        return pos

    encode = encode_boxes  # Backwards compatibility

    @torch.no_grad()
    def encode_points(self, x, y, labels):
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        pos = torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)
        return pos

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        cache_key = (x.shape[-2], x.shape[-1])
        if cache_key in self.cache:
            return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
        y_embed = (
            torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])
        )
        x_embed = (
            torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)
        )

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]
        return pos


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


# Rotary Positional Encoding, adapted from:
# 1. https://github.com/meta-llama/codellama/blob/main/llama/model.py
# 2. https://github.com/naver-ai/rope-vit
# 3. https://github.com/lucidrains/rotary-embedding-torch

def init_t_xy(end_x: int, end_y: int, device=None):
    t = torch.arange(end_x * end_y, dtype=torch.float32, device=device)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y

def compute_axial_rope_cos_sin(dim: int, end_x: int, end_y: int, theta: float = 10000.):
    # dim: 需要能被4整除
    assert dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs)  # [end_x*end_y, dim//2]
    freqs_y = torch.outer(t_y, freqs)  # [end_x*end_y, dim//2]
    # 拼在一起
    freqs = torch.cat([freqs_x, freqs_y], dim=-1)  # [seq_len, dim]
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin  # [seq_len, dim]

def reshape_for_broadcast(param, x):
    # param: [seq_len, dim], x: [..., seq_len, dim]
    # reshape param为[1, ..., seq_len, dim]
    ndim = x.ndim
    shape = [1]*(ndim-2) + list(param.shape)
    return param.view(*shape)

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # x: [B, n_head, seq_len, head_dim]  head_dim必须是偶数
    # cos/sin: [seq_len, head_dim]
    # 广播到x形状
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos, sin = cos[..., ::2], sin[..., ::2]
    x_out_even = x1 * cos - x2 * sin
    x_out_odd  = x1 * sin + x2 * cos
    x_out = torch.stack([x_out_even, x_out_odd], dim=-1)
    x_out = x_out.flatten(-2)  # 恢复到原始最后一维
    return x_out



# Matrix version of rotary enc
# https://github.com/facebookresearch/segment-anything-2/issues/186

def get_rotation_matrices(dim, end_x, end_y, theta=10000.0, device=None, dtype=None):
    
    powers = torch.linspace(0, 1, 1 + (dim // 4), device=device, dtype=dtype)[:-1]
    base_angles = torch.pow(theta, -powers)

    end_x, end_y = int(end_x), int(end_y)
    x_mults = torch.arange(end_x, device=device, dtype=dtype).repeat(end_y)
    y_mults = torch.arange(end_y, device=device, dtype=dtype).repeat_interleave(end_x)
    angles_xy = (torch.outer(mults, base_angles) for mults in (x_mults, y_mults))
    
    rotmats_list = []
    for angles in angles_xy:
        sterm, cterm = torch.sin(-angles), torch.cos(-angles)
        rotmat = torch.stack(
            [
                torch.stack([cterm, -sterm], dim=-1),
                torch.stack([sterm, cterm], dim=-1),
            ],
            dim=-1,
        )
        rotmats_list.append(rotmat)

    return torch.cat(rotmats_list, dim=1).unsqueeze(0).unsqueeze(0)


def apply_rotary_matenc(xq, xk, rotmats, repeat_freqs_k=False):
    
    bq, hq, nq, cq = xq.shape
    bk, hk, nk, ck = xk.shape

    q_out = torch.matmul(rotmats, xq.reshape(bq, hq, nq, cq // 2, 2, 1)).flatten(3)
    k_rotmat = rotmats.repeat(1, 1, nk // nq, 1, 1, 1) if repeat_freqs_k else rotmats
    k_out = torch.matmul(k_rotmat, xk.reshape(bk, hk, nk, ck // 2, 2, 1)).flatten(3)

    return q_out, k_out
