# models/fusion/double_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

# ========== 诊断工具 ==========
class AttentionExtremesDiagnosis:
    """
    注意力极端化问题诊断器：监控注意力分布（简单统计），便于训练期调参
    """
    def analyze_attention_distribution(self, attention_weights, layer_name: str):
        # attention_weights: [B, H, Q, K]
        with torch.no_grad():
            B, H, Q, K = attention_weights.shape
            metrics = {}
            head = 0
            w = attention_weights[0, head]  # [Q, K]
            max_attn = w.max(dim=-1).values.mean()
            min_attn = w.min(dim=-1).values.mean()
            p = w.clamp_min(1e-8)
            entropy = -(p * p.log()).sum(dim=-1).mean()
            max_entropy = torch.log(torch.tensor(K, device=w.device, dtype=w.dtype))
            norm_entropy = (entropy / max_entropy).item()
            sparsity = (w < 0.1).float().mean().item()
            if norm_entropy > 0.9 and max_attn < 0.3:
                diag = "过度分散"
            elif norm_entropy < 0.1 and max_attn > 0.8 and sparsity > 0.8:
                diag = "过度集中"
            else:
                diag = "正常范围"
            metrics["head_0"] = {
                "max_attention": float(max_attn),
                "min_attention": float(min_attn),
                "entropy": float(norm_entropy),
                "sparsity": float(sparsity),
                "diagnosis": diag,
            }
            return metrics


# ========== 交叉注意力 ==========
class StabilizedCrossAttention(nn.Module):
    """
    稳定的交叉注意力层（带相对位置偏置）
    """
    def __init__(self, hidden_size, num_heads, dropout=0.1, max_relative_positions=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_relative_positions = max_relative_positions

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.relative_attention_bias = nn.Embedding(
            2 * max_relative_positions - 1, num_heads
        )
        self.dropout = nn.Dropout(dropout)

        self.attention_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.local_bias_matrix = self._create_local_bias(2048, window_size=5)

        self.diagnosis = AttentionExtremesDiagnosis()

    def _create_local_bias(self, max_len, window_size):
        bias = torch.zeros(max_len, max_len, dtype=torch.float16)
        for i in range(max_len):
            s = max(0, i - window_size)
            e = min(max_len, i + window_size + 1)
            bias[i, s:e] = 1.0
        return nn.Parameter(bias, requires_grad=False)

    def _compute_relative_positions(self, q_len, k_len, device):
        rq = torch.arange(q_len, device=device)
        rk = torch.arange(k_len, device=device)
        dist = rk[None, :] - rq[:, None]
        clipped = torch.clamp(
            dist + self.max_relative_positions - 1, 0,
            2 * self.max_relative_positions - 2
        )
        return clipped

    def forward(self, query, key, value):
        """
        query: [B, Q, H], key/value: [B, K, H]
        return: attn_out [B,Q,H], attn_weights [B,H,Q,K], metrics
        """
        B, Q, _ = query.shape
        _, K, _ = key.shape

        q = self.q_proj(query).view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        rel_pos = self._compute_relative_positions(Q, K, query.device)
        rel_bias = self.relative_attention_bias(rel_pos)  # [Q,K,H]
        rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0).to(attn_scores.dtype)
        attn_scores = attn_scores + rel_bias

        if Q <= self.local_bias_matrix.size(0) and K <= self.local_bias_matrix.size(0):
            local_bias = self.local_bias_matrix[:Q, :K].unsqueeze(0).unsqueeze(0).to(attn_scores.dtype)
            attn_scores = attn_scores + local_bias * 0.05

        scale = torch.exp(self.log_scale.to(attn_scores.dtype)).clamp(min=1.0, max=4.0)
        attn_scores = attn_scores * scale + self.attention_bias.to(attn_scores.dtype)

        attn_weights = F.softmax(attn_scores, dim=-1)
        metrics = self.diagnosis.analyze_attention_distribution(
            attn_weights.clamp_min(1e-8), "cross_attention"
        )
        attn_weights = self.dropout(attn_weights).to(v.dtype)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, Q, self.hidden_size)
        out = self.out_proj(out)
        return out, attn_weights, metrics


# ========== 自注意力 ==========使用因果掩码的（但是对于压缩器/融合器最好是非因果）
class DiversifiedSelfAttention(nn.Module):
    """
    多样化自注意力
    """
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.diagnosis = AttentionExtremesDiagnosis()

    def forward(self, x):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        causal_mask = torch.full(
            (L, L), float("-inf"), dtype=attn_scores.dtype, device=x.device
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)
        attn_scores = torch.clamp(attn_scores, min=-50.0, max=50.0)

        attn_weights = F.softmax(attn_scores, dim=-1)
        metrics = self.diagnosis.analyze_attention_distribution(
            attn_weights.clamp_min(1e-8), "self_attention"
        )
        attn_weights = self.dropout(attn_weights).to(v.dtype)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        out = self.out_proj(out)
        return out, attn_weights, metrics


# ========== 约束 ==========
class CrossAttentionConstraint(nn.Module):
    """交叉注意力约束：简单的熵与最大注意力惩罚"""
    def __init__(self, diversity_weight=0.1, max_attention_weight=0.2):
        super().__init__()
        self.diversity_weight = diversity_weight
        self.max_attention_weight = max_attention_weight

    def forward(self, attention_weights):
        _, _, _, K = attention_weights.shape
        p = attention_weights.clamp_min(1e-8)
        entropy = -(p * p.log()).sum(dim=-1).mean()
        target = torch.log(torch.tensor(K, device=p.device, dtype=p.dtype)) * 0.6
        diversity_loss = F.mse_loss(entropy, target)

        max_attn = attention_weights.max(dim=-1).values
        max_penalty = torch.relu(max_attn - 0.8).mean()

        loss = self.diversity_weight * diversity_loss + self.max_attention_weight * max_penalty
        return loss.to(attention_weights.dtype)


class SelfAttentionConstraint(nn.Module):
    """自注意力的本地性约束（目前没在训练脚本里用）"""
    def __init__(self, local_global_weight=0.1, window_size=3):
        super().__init__()
        self.local_global_weight = local_global_weight
        self.window_size = window_size

    def forward(self, attention_weights):
        B, H, L, _ = attention_weights.shape
        mask = attention_weights.new_zeros((L, L))
        for i in range(L):
            s, e = max(0, i - self.window_size), min(L, i + self.window_size + 1)
            mask[i, s:e] = 1.0
        mask = mask.unsqueeze(0).unsqueeze(0)
        local = (attention_weights * mask).sum(dim=(-1, -2)) / attention_weights.sum(dim=(-1, -2))
        target = attention_weights.new_tensor(0.4)
        loss = F.mse_loss(local.mean(), target)
        return loss


# ========== 双层注意力层 ==========
class DoubleAttentionLayer(nn.Module):
    """
    1) Cross-Attn  (Q=question, K/V=context)
    2) Residual
    3) Self-Attn
    4) Residual + LayerNorm
    """
    def __init__(self, hidden_dim=4096, num_heads=16, dropout=0.1):
        super().__init__()
        self.cross_attention = StabilizedCrossAttention(hidden_dim, num_heads, dropout)
        self.self_attention = DiversifiedSelfAttention(hidden_dim, num_heads, dropout)
        self.self_out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ca_constraint = CrossAttentionConstraint(0.1, 0.2)

    def forward(self, question_features, context_features):
        cross_out, cross_w, cross_m = self.cross_attention(
            question_features, context_features, context_features
        )
        resid1 = cross_out + question_features

        self_out, self_w, self_m = self.self_attention(resid1)
        self_out = self.self_out_proj(self_out)
        self_out = self.norm(self_out)
        resid2 = resid1 + self_out
        final = resid2

        constraint_loss = self.ca_constraint(cross_w)
        info = {
            "cross_metrics": cross_m,
            "cross_weights": cross_w,
            "self_metrics": self_m,
            "self_weights": self_w,
            "constraint_loss": constraint_loss,
        }
        return final, info


class DoubleAttentionStack(nn.Module):
    """
    堆叠 L 层 DoubleAttentionLayer（上下文 K/V 固定为同一份 context_emb）
    对外返回 summary + 最后一层 attention_info + 累积约束损失
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        self.layers = nn.ModuleList([
            DoubleAttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, question_features: torch.Tensor, context_features: torch.Tensor):
        summary = question_features
        total_constraint = None
        last_info = None

        for layer in self.layers:
            summary, info = layer(summary, context_features)
            if total_constraint is None:
                total_constraint = info["constraint_loss"].clone()
            else:
                total_constraint = total_constraint + info["constraint_loss"]
            last_info = info

        attention_info = {
            "cross_metrics": last_info["cross_metrics"],
            "cross_weights": last_info["cross_weights"],
            "self_metrics":  last_info["self_metrics"],
            "self_weights":  last_info["self_weights"],
            "constraint_loss": total_constraint,
        }
        return summary, attention_info

