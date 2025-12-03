
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

from double_attention import StabilizedCrossAttention

# StabilizedCrossAttention / chunk_pooling_scores 定义见 double_attention.py :contentReference[oaicite:1]{index=1}


def chunk_pooling_scores(
        cross_weights: torch.Tensor,
        chunk_spans: List[List[Tuple[int, int]]],
        reduce_heads: bool = True,
        reduce_query: bool = True,
        normalize: bool = True,
    ) -> List[torch.Tensor]:
    """
    cross_weights: [B,H,Q,K] （来自最后一层）
    chunk_spans: 每个 batch 一组 [(s,e), ...]，e 为开区间
    返回：List[Tensor[J]]，每个样本一个 [J] 分数向量
    """
    # 1) 头平均
    attn = cross_weights
    if reduce_heads:
        attn = attn.mean(dim=1)  # [B, Q, K]
    # 2) query 平均
    if reduce_query:
        attn = attn.mean(dim=1)  # [B, K]
    B, K = attn.shape
    out_scores: List[torch.Tensor] = []

    for b in range(B):
        scores_b = []
        for (s, e) in chunk_spans[b]:
            s_clamped = max(0, min(K, s))
            e_clamped = max(s_clamped, min(K, e))
            if e_clamped > s_clamped:
                # 先 chunk 内平均，避免长 chunk 直接吃更多权重
                score = attn[b, s_clamped:e_clamped].mean()
            else:
                score = attn.new_tensor(0.0)
            scores_b.append(score)
        scores_b = torch.stack(scores_b) if scores_b else attn.new_zeros(0)
        if normalize and scores_b.numel() > 0:
            scores_b = torch.softmax(scores_b, dim=0)
        out_scores.append(scores_b)
    return out_scores

class LocalChunkScorer(nn.Module):
    """
    AttentionRAG 风格的局部检索：
    - 每个 chunk 单独做一次 cross-attn，Q=question, K/V=该 chunk
    - 对注意力做简单 pooling 得到一个标量分数
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = StabilizedCrossAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        question_emb: torch.Tensor,               # [B,Q,H]
        context_emb: torch.Tensor,                # [B,K,H]
        chunk_spans: List[List[Tuple[int, int]]],
        normalize: bool = True,
    ) -> List[torch.Tensor]:
        """
        返回：List[Tensor[J_b]]，每个样本一条 chunk 分数向量。
        注意：这里是“局部视角”，每个 chunk 独立看，不考虑其它 chunk 的干扰。
        """
        B, Q, H = question_emb.shape
        _, K, _ = context_emb.shape
        device = question_emb.device
        scores_list: List[torch.Tensor] = []

        for b in range(B):
            spans_b = chunk_spans[b] if b < len(chunk_spans) else []
            if not spans_b:
                scores_list.append(context_emb.new_zeros(0))
                continue

            q_b = question_emb[b:b+1]   # [1,Q,H]
            scores_b = []
            for (s, e) in spans_b:
                s_clamped = max(0, min(K, s))
                e_clamped = max(s_clamped, min(K, e))
                if e_clamped <= s_clamped:
                    scores_b.append(context_emb.new_tensor(0.0))
                    continue
                c_chunk = context_emb[b:b+1, s_clamped:e_clamped, :]  # [1,Lc,H]

                # 单 chunk cross-attn
                _, attn_w, _ = self.cross_attn(
                    query=q_b,   # [1,Q,H]
                    key=c_chunk, # [1,Lc,H]
                    value=c_chunk,
                )  # attn_w: [1,H,Q,Lc]

                # # 池化：mean over heads, query, tokens -> 标量
                # w = attn_w.mean(dim=1).mean(dim=1).mean(dim=1)  # [1]
                # scores_b.append(w.squeeze(0))

                # AttentionRAG 风格池化：
                # 1) 先对 token 做 max-pooling：max over Lc
                # 2) 再对 head / query 做平均：mean over H, Q
                # 得到一个标量分数
                w = attn_w.max(dim=-1).values        # [1,H,Q]
                w = w.mean(dim=1).mean(dim=1)        # [1]
                scores_b.append(w.squeeze(0))

            scores_b = torch.stack(scores_b, dim=0)  # [J_b]
            if normalize and scores_b.numel() > 0:
                scores_b = torch.softmax(scores_b, dim=0)
            scores_list.append(scores_b)

        return scores_list


class GlobalChunkScorer(nn.Module):
    """
    全局检索：
    - 把所有 chunk 拼在一起，用一遍 cross-attn
    - 通过 chunk_pooling_scores 在 chunk 级别做 pooling
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = StabilizedCrossAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        question_emb: torch.Tensor,               # [B,Q,H]
        context_emb: torch.Tensor,                # [B,K,H]
        chunk_spans: List[List[Tuple[int, int]]],
        normalize: bool = True,
    ) -> List[torch.Tensor]:
        """
        返回：List[Tensor[J_b]]，每个样本一条 chunk 分数向量。
        这里“全局视角”，所有 chunk 一起参与 cross-attn。
        """
        B, Q, H = question_emb.shape
        _, K, _ = context_emb.shape

        cross_out, cross_weights, _ = self.cross_attn(
            query=question_emb,
            key=context_emb,
            value=context_emb,
        )  # cross_weights: [B,H,Q,K]

        scores = chunk_pooling_scores(
            cross_weights=cross_weights,
            chunk_spans=chunk_spans,
            reduce_heads=True,
            reduce_query=True,
            normalize=normalize,
        )
        return scores


class MultiViewEvidenceRouter(nn.Module):
    """
    Multi-view router:
    - 根据 question / chunk 语义，对 local / global scores 做门控融合。
    - 同时产出一个 global_gate，供后续 SSM / 全局模块使用（这里先返回，不使用）。
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def _get_chunk_vec(
        self,
        context_emb_b: torch.Tensor,      # [K,H]
        span: Tuple[int, int],
    ) -> torch.Tensor:
        s, e = span
        K, H = context_emb_b.shape
        s = max(0, min(K, s))
        e = max(s, min(K, e))
        if e <= s:
            return context_emb_b.new_zeros(H)
        chunk = context_emb_b[s:e]  # [L_j,H]
        return self.layer_norm(chunk.mean(dim=0))  # [H]

    def forward(
        self,
        question_emb: torch.Tensor,            # [B,Q,H]
        context_emb: torch.Tensor,             # [B,K,H]
        local_scores: List[torch.Tensor],      # [J_b]
        global_scores: List[torch.Tensor],     # [J_b]
        chunk_spans: List[List[Tuple[int,int]]],
    ):
        B, Q, H = question_emb.shape
        final_scores: List[torch.Tensor] = []
        global_gates: List[torch.Tensor] = []

        for b in range(B):
            spans_b = chunk_spans[b] if b < len(chunk_spans) else []
            if not spans_b:
                final_scores.append(context_emb.new_zeros(0))
                global_gates.append(context_emb.new_tensor([[1.0]]))
                continue

            ls_b = local_scores[b]
            gs_b = global_scores[b]
            if ls_b.numel() == 0 or gs_b.numel() == 0:
                final_scores.append(context_emb.new_zeros(0))
                global_gates.append(context_emb.new_tensor([[1.0]]))
                continue

            # 对问题取一个全局向量
            q_vec = self.layer_norm(question_emb[b].mean(dim=0))  # [H]

            gates = []
            for span in spans_b:
                chunk_vec = self._get_chunk_vec(context_emb[b], span)  # [H]
                g_in = torch.cat([q_vec, chunk_vec], dim=-1)           # [2H]
                g = self.gate_proj(g_in)                               # [1]
                gates.append(g)

            gates = torch.cat(gates, dim=0)  # [J_b]
            fused = gates * ls_b + (1.0 - gates) * gs_b  # [J_b]

            if fused.numel() > 0:
                fused = torch.softmax(fused, dim=0)

            final_scores.append(fused)
            global_gates.append(fused.mean().view(1, 1))

        global_gate = torch.stack(global_gates, dim=0)  # [B,1,1]
        return final_scores, global_gate


class MultiViewChunkRetrieval(nn.Module):
    """
    评估器入口：
    - LocalChunkScorer: 局部
    - GlobalChunkScorer: 全局
    - MultiViewEvidenceRouter: 融合 local/global
    输出：
    - final_scores: 用于剪枝 / loss
    - local_scores, global_scores: 方便做 ablation / 辅助损失
    - global_gate: 供后续模块使用（本版本模型暂未使用）
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.local_scorer = LocalChunkScorer(hidden_size, num_heads, dropout)
        self.global_scorer = GlobalChunkScorer(hidden_size, num_heads, dropout)
        self.router = MultiViewEvidenceRouter(hidden_size)

    def forward(
        self,
        question_emb: torch.Tensor,               # [B,Q,H]
        context_emb: torch.Tensor,                # [B,K,H]
        chunk_spans: List[List[Tuple[int,int]]],
    ) -> Dict[str, object]:
        local_scores = self.local_scorer(question_emb, context_emb, chunk_spans)
        global_scores = self.global_scorer(question_emb, context_emb, chunk_spans)
        final_scores, global_gate = self.router(
            question_emb=question_emb,
            context_emb=context_emb,
            local_scores=local_scores,
            global_scores=global_scores,
            chunk_spans=chunk_spans,
        )
        return {
            "final_scores": final_scores,
            "local_scores": local_scores,
            "global_scores": global_scores,
            "global_gate": global_gate,
        }

