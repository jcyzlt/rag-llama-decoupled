# latent_prefix_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List

from double_attention import DiversifiedSelfAttention, CrossAttentionConstraint


class MaskedStabilizedCrossAttention(nn.Module):
    """
    基于你原始 StabilizedCrossAttention 的实现重写：
    - 参数名/结构保持一致
    - 增加 key_padding_mask: [B, K], 1表示有效，0表示pad
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1, max_relative_positions: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_relative_positions = max_relative_positions

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # 相对位置偏置（与你原实现一致）
        self.relative_attention_bias = nn.Embedding(2 * max_relative_positions - 1, num_heads)

        # 简单 local bias（与你原实现一致的“局部偏置矩阵”思路）
        self.local_bias_matrix = nn.Parameter(torch.zeros(512, 512), requires_grad=False)

        self.log_scale = nn.Parameter(torch.zeros(1))
        self.attention_bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)

    def _relative_positions_bucket(self, relative_positions: torch.Tensor):
        # relative_positions: [Q, K]
        # 与你原实现一致：截断到 [-max+1, max-1] 后映射到 [0, 2*max-2]
        max_rel = self.max_relative_positions
        clipped = relative_positions.clamp(min=-(max_rel - 1), max=(max_rel - 1))
        bucket = clipped + (max_rel - 1)
        return bucket

    def forward(
        self,
        query: torch.Tensor,                    # [B, Q, H]
        key: torch.Tensor,                      # [B, K, H]
        value: torch.Tensor,                    # [B, K, H]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, K] 1有效 0pad
    ):
        B, Q, _ = query.shape
        _, K, _ = key.shape

        q = self.q_proj(query).view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,Q,D]
        k = self.k_proj(key).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)    # [B,H,K,D]
        v = self.v_proj(value).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,K,D]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)        # [B,H,Q,K]

        # 相对位置偏置
        q_pos = torch.arange(Q, device=query.device)
        k_pos = torch.arange(K, device=query.device)
        rel = q_pos[:, None] - k_pos[None, :]                                               # [Q,K]
        rp_bucket = self._relative_positions_bucket(rel)                                     # [Q,K]
        rel_bias = self.relative_attention_bias(rp_bucket).permute(2, 0, 1)                  # [H,Q,K]
        attn_scores = attn_scores + rel_bias.unsqueeze(0).to(attn_scores.dtype)

        # local bias（可选：保持与你原文件类似的逻辑）
        if Q <= self.local_bias_matrix.size(0) and K <= self.local_bias_matrix.size(1):
            local_bias = self.local_bias_matrix[:Q, :K].unsqueeze(0).unsqueeze(0).to(attn_scores.dtype)
            attn_scores = attn_scores + local_bias * 0.05

        # scale/bias（与你原实现一致）
        scale = torch.exp(self.log_scale.to(attn_scores.dtype)).clamp(min=1.0, max=4.0)
        attn_scores = attn_scores * scale + self.attention_bias.to(attn_scores.dtype)

        # ✅ 关键：mask 掉 padding 的 key 位置
        if key_padding_mask is not None:
            # key_padding_mask: 1有效 0pad -> pad位置置为 -inf
            pad = (key_padding_mask == 0).unsqueeze(1).unsqueeze(2)  # [B,1,1,K]
            attn_scores = attn_scores.masked_fill(pad, float("-inf"))

        attn_scores = torch.clamp(attn_scores, min=-50.0, max=50.0)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B,H,Q,K]
        attn_weights = self.dropout(attn_weights).to(v.dtype)

        out = torch.matmul(attn_weights, v)  # [B,H,Q,D]
        out = out.transpose(1, 2).contiguous().view(B, Q, self.hidden_size)
        out = self.out_proj(out)
        return out, attn_weights


class LatentPrefixAttentionLayer(nn.Module):
    """
    单层 Latent Prefix:
      - latent slots: [B,M,H] 作为 Query
      - KV: context 或 [question; context]
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        include_question_in_kv: bool = True,
        max_relative_positions: int = 512,
    ):
        super().__init__()
        self.include_question_in_kv = include_question_in_kv

        self.cross_attn = MaskedStabilizedCrossAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            max_relative_positions=max_relative_positions,
        )

        # 你原 Self-Attn 固定 causal mask；latent prefix 没 padding，可直接用
        self.self_attn = DiversifiedSelfAttention(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)

        self.ca_constraint = CrossAttentionConstraint(0.1, 0.2)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        latent: torch.Tensor,                   # [B,M,H]
        question: torch.Tensor,                 # [B,Q,H]
        context: torch.Tensor,                  # [B,K,H]
        question_mask: Optional[torch.Tensor] = None,  # [B,Q] 1有效 0pad（通常全1）
        context_mask: Optional[torch.Tensor] = None,   # [B,K] 1有效 0pad
    ):
        if self.include_question_in_kv:
            kv = torch.cat([question, context], dim=1)
            if question_mask is not None and context_mask is not None:
                kv_mask = torch.cat([question_mask, context_mask], dim=1)
            else:
                kv_mask = None
        else:
            kv, kv_mask = context, context_mask

        cross_out, attn_w = self.cross_attn(latent, kv, kv, key_padding_mask=kv_mask)
        latent = self.norm1(latent + self.dropout(cross_out))

        self_out,_,_ = self.self_attn(latent)
        latent = self.norm2(latent + self.dropout(self_out))

        constraint_loss = self.ca_constraint(attn_w)
        return latent, {"cross_weights":attn_w,"constraint_loss":constraint_loss}


class LatentPrefixAttentionStack(nn.Module):
    """
    多层 latent prefix stack：输出固定长度 prefix=[B,M,H]
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        prefix_len: int,
        dropout: float = 0.1,
        include_question_in_kv: bool = True,
        max_relative_positions: int = 512,
    ):
        super().__init__()
        self.prefix_len = prefix_len
        self.latent_slots = nn.Parameter(torch.randn(1, prefix_len, hidden_size) * 0.02)

        self.layers = nn.ModuleList([
            LatentPrefixAttentionLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                include_question_in_kv=include_question_in_kv,
                max_relative_positions=max_relative_positions,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        question: torch.Tensor,                       # [B,Q,H]
        context: torch.Tensor,                        # [B,K,H]
        question_mask: Optional[torch.Tensor] = None, # [B,Q]
        context_mask: Optional[torch.Tensor] = None,  # [B,K]
    ):
        B = question.size(0)
        latent = self.latent_slots.expand(B, -1, -1)

        total_constraint = 0.0
        last_cross_weights = None  # 新增
        for layer in self.layers:
            latent, layer_info = layer(latent, question, context, question_mask, context_mask)
            total_constraint = total_constraint + layer_info["constraint_loss"]
            last_cross_weights = layer_info.get("cross_weights")  # 捕获

        return latent, {"constraint_loss": total_constraint,"cross_weights": last_cross_weights}
