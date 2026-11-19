from math import sqrt

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class GeometricReasoningOriginalImpl(nn.Module):
    def __init__(
        self,
        c_s: int,
        v_heads: int,
        num_vector_messages: int = 1,
        mask_and_zero_frameless: bool = True,
        divide_residual_by_depth: bool = False,
        bias: bool = False,
    ):
        """Approximate implementation:

        ATTN(A, v) := (softmax_j A_ij) v_j
        make_rot_vectors(x) := R(i->g) Linear(x).reshape(..., 3)
        make_vectors(x) := T(i->g) Linear(x).reshape(..., 3)

        v <- make_rot_vectors(x)
        q_dir, k_dir <- make_rot_vectors(x)
        q_dist, k_dist <- make_vectors(x)

        A_ij       <- dot(q_dir_i, k_dir_j) -||q_dist_i - k_dist_j||^2
        x          <- x + Linear(T(g->i) ATTN(A, v))
        """
        super().__init__()
        self.c_s = c_s
        self.v_heads = v_heads
        self.num_vector_messages = num_vector_messages
        self.mask_and_zero_frameless = mask_and_zero_frameless

        self.s_norm = nn.LayerNorm(c_s, bias=bias)
        dim_proj = (
            4 * self.v_heads * 3 + self.v_heads * 3 * self.num_vector_messages
        )  # 2 x (q, k) * number of heads * (x, y, z) + number of heads * number of vector messages * (x, y, z)
        self.proj = nn.Linear(c_s, dim_proj, bias=bias)
        channels_out = self.v_heads * 3 * self.num_vector_messages
        self.out_proj = nn.Linear(channels_out, c_s, bias=bias)

        # The basic idea is for some attention heads to pay more or less attention to rotation versus distance,
        # as well as to control the sharpness of the softmax (i.e., should this head only attend to those residues
        # very nearby or should there be shallower dropoff in attention weight?)
        self.distance_scale_per_head = nn.Parameter(torch.zeros((self.v_heads)))
        self.rotation_scale_per_head = nn.Parameter(torch.zeros((self.v_heads)))

    def forward(self, s, affine, affine_mask, sequence_id, chain_id):
        if sequence_id is None:
            sequence_id = torch.zeros_like(s[..., 0], dtype=torch.int64)
        attn_bias = sequence_id.unsqueeze(-1) == sequence_id.unsqueeze(-2)
        attn_bias = attn_bias.unsqueeze(1).float()
        attn_bias = attn_bias.masked_fill(
            ~affine_mask[:, None, None, :], torch.finfo(attn_bias.dtype).min
        )
        chain_id_mask = chain_id.unsqueeze(1) != chain_id.unsqueeze(2)
        attn_bias = attn_bias.masked_fill(
            chain_id_mask.unsqueeze(1), torch.finfo(s.dtype).min
        )

        ns = self.s_norm(s)
        vec = self.proj(ns).float()
        vec_rot, vec_dist = vec.split(
            [
                self.v_heads * 2 * 3 + self.v_heads * 3 * self.num_vector_messages,
                self.v_heads * 2 * 3,
            ],
            dim=-1,
        )

        # Rotate the queries and keys for the rotation term. We also rotate the values.
        # NOTE(zeming, thayes): Values are only rotated, not translated. We may wish to change
        # this in the future.
        query_rot, key_rot, value = (
            affine.rot[..., None]
            .apply(rearrange(vec_rot, "... (h c) -> ... h c", c=3))
            .split(
                [self.v_heads, self.v_heads, self.v_heads * self.num_vector_messages],
                dim=-2,
            )
        )

        # Rotate and translate the queries and keys for the distance term
        # NOTE(thayes): a simple speedup would be to apply all rotations together, then
        # separately apply the translations.
        query_dist, key_dist = (
            affine[..., None]
            .apply(rearrange(vec_dist, "... (h c) -> ... h c", c=3))
            .chunk(2, dim=-2)
        )

        query_dist = rearrange(query_dist, "b s h d -> b h s 1 d")
        key_dist = rearrange(key_dist, "b s h d -> b h 1 s d")
        query_rot = rearrange(query_rot, "b s h d -> b h s d")
        key_rot = rearrange(key_rot, "b s h d -> b h d s")
        value = rearrange(
            value, "b s (h m) d -> b h s (m d)", m=self.num_vector_messages
        )

        distance_term = (query_dist - key_dist).norm(dim=-1) / sqrt(3)
        rotation_term = query_rot.matmul(key_rot) / sqrt(3)
        distance_term_weight = rearrange(
            F.softplus(self.distance_scale_per_head), "h -> h 1 1"
        )
        rotation_term_weight = rearrange(
            F.softplus(self.rotation_scale_per_head), "h -> h 1 1"
        )

        attn_weight = (
            rotation_term * rotation_term_weight - distance_term * distance_term_weight
        )

        if attn_bias is not None:
            # we can re-use the attention bias from the transformer layers
            # NOTE(thayes): This attention bias is expected to handle two things:
            # 1. Masking attention on padding tokens
            # 2. Masking cross sequence attention in the case of bin packing
            s_q = attn_weight.size(2)
            s_k = attn_weight.size(3)
            _s_q = max(0, attn_bias.size(2) - s_q)
            _s_k = max(0, attn_bias.size(3) - s_k)
            attn_bias = attn_bias[:, :, _s_q:, _s_k:]
            attn_weight = attn_weight + attn_bias

        attn_weight = torch.softmax(attn_weight, dim=-1)

        attn_out = attn_weight.matmul(value)

        attn_out = (
            affine.rot[..., None]
            .invert()
            .apply(
                rearrange(
                    attn_out, "b h s (m d) -> b s (h m) d", m=self.num_vector_messages
                )
            )
        )

        attn_out = rearrange(
            attn_out, "b s (h m) d -> b s (h m d)", m=self.num_vector_messages
        )
        if self.mask_and_zero_frameless:
            attn_out = attn_out.masked_fill(~affine_mask[..., None], 0.0)
        s = self.out_proj(attn_out.to(s.dtype))

        return s
