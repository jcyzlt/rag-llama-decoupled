# models/losses/attention_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFocusLoss(nn.Module):
    """简单的注意力熵正则"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, attention_weights):
        p = attention_weights.clamp_min(1e-8)
        entropy = -(p * p.log()).sum(dim=-1)
        return entropy.mean()


class AlignmentLoss(nn.Module):
    """问题-上下文对齐损失（Top-k 余弦距离）"""
    def __init__(self, topk: int = 5):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=-1)
        self.topk = topk

    def forward(self, question_emb, context_emb, cross_attn_weights):
        # question_emb: [B,Q,H]; context_emb: [B,K,H]; cross_attn_weights: [B,H,Q,K]
        avg_attn = cross_attn_weights.mean(dim=(1, 2))  # [B,K]
        k = min(self.topk, context_emb.size(1))
        _, idx = torch.topk(avg_attn, k=k, dim=-1)
        loss = 0.0
        for b in range(context_emb.size(0)):
            top_ctx = context_emb[b, idx[b]]         # [k,H]
            q_vec = question_emb[b].mean(dim=0, keepdim=True)  # [1,H]
            sim = self.cos(q_vec, top_ctx)
            loss = loss + (1 - sim).mean()
        return loss / context_emb.size(0)


class ProgressiveConstraintTraining:
    """
    渐进式约束调度：基于 cross_metrics["entropy"] 做简单区间约束
    """
    def __init__(self, total_steps: int):
        self.total_steps = max(1, int(total_steps))
        self.current_step = 0
        self.base_min = 2.0
        self.base_max = 4.0

    def get_current_constraints(self):
        progress = min(1.0, max(0.0, self.current_step / self.total_steps))
        decay = 1.0 - 0.8 * progress
        cur_min = self.base_min * (0.2 + 0.8 * decay)
        cur_max = self.base_max * (0.8 + 0.2 * decay)
        return cur_min, cur_max

    def compute_constraint_loss(self, cross_metrics: dict) -> float:
        ent = cross_metrics.get("head_0", {}).get("entropy", 0.0)
        try:
            ent = float(ent)
        except Exception:
            ent = 0.0
        cmin, cmax = self.get_current_constraints()
        if ent < cmin:
            loss = (cmin - ent) ** 2
        elif ent > cmax:
            loss = (ent - cmax) ** 2
        else:
            loss = 0.0
        return float(loss)

    def step(self):
        self.current_step += 1
