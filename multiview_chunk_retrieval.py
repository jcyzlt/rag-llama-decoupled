
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



# class LocalChunkScorer(nn.Module):
#     """
#     AttentionRAG 风格的局部检索（小 Transformer + 注意力特征 + MLP 打分）：
#     - 对每个 chunk，构造 [question; chunk] 序列
#     - 用若干层 StabilizedCrossAttention 当 self-attention 编码该序列
#     - 从最后一层 self-attn 中取 question→chunk 的注意力，做 pooling 得到一个标量 attention 特征
#     - 再用编码后的向量 + attention 特征，交给 MLP 输出 chunk logit（注意：这里输出的是 logits，不做 softmax）
#     """
#     def __init__(
#         self,
#         hidden_size: int,
#         num_heads: int,
#         dropout: float = 0.1,
#         num_layers: int = 1,
#         use_attn_feature: bool = True,
#     ):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         self.use_attn_feature = use_attn_feature

#         # 小型 Transformer encoder：堆几层 StabilizedCrossAttention，当 self-attn 用
#         self.layers = nn.ModuleList([
#             nn.ModuleDict({
#                 "attn":StabilizedCrossAttention(
#                     hidden_size=hidden_size,
#                     num_heads=num_heads,
#                     dropout=dropout,),
#                 "norm":nn.LayerNorm(hidden_size),
#             })
#             for _ in range(num_layers)
#         ])
#         # self.norm = nn.LayerNorm(hidden_size)

#         # MLP 打分头：输入 = [q_repr, c_repr, q_repr⊙c_repr, attn_scalar]
#         attn_feat_dim = 1 if use_attn_feature else 0
#         in_dim = 3 * hidden_size + attn_feat_dim

#         self.score_head = nn.Sequential(
#             nn.Linear(in_dim, hidden_size),
#             nn.GELU(),
#             nn.Linear(hidden_size, 1),
#         )

#     def forward(
#         self,
#         question_emb: torch.Tensor,               # [B,Q,H]  这里通常是 LLaMA 的 embedding / 中间层输出
#         context_emb: torch.Tensor,                # [B,K,H]
#         chunk_spans: List[List[Tuple[int, int]]],
#         normalize: bool = False,                  # 注意：这里默认返回 logits，不做 softmax
#     ) -> List[torch.Tensor]:
#         """
#         返回：
#         - scores_list: List[Tensor[J_b]]，每个样本一条 chunk 分数向量（logits）
#         """
#         B, Q, H = question_emb.shape
#         _, K, _ = context_emb.shape
#         scores_list: List[torch.Tensor] = []

#         for b in range(B):
#             spans_b = chunk_spans[b] if b < len(chunk_spans) else []
#             if not spans_b:
#                 scores_list.append(context_emb.new_zeros(0))
#                 continue

#             q_b = question_emb[b:b+1]    # [1,Q,H]
#             scores_b = []

#             for (s, e) in spans_b:
#                 # 保守 clamp 一下 span，防越界
#                 s_clamped = max(0, min(K, s))
#                 e_clamped = max(s_clamped, min(K, e))
#                 if e_clamped <= s_clamped:
#                     scores_b.append(context_emb.new_tensor(0.0))
#                     continue

#                 # 取该 chunk 的 token 表示
#                 c_chunk = context_emb[b:b+1, s_clamped:e_clamped, :]   # [1,Lc,H]
#                 Lc = e_clamped - s_clamped

#                 # [question; chunk] 序列
#                 seq = torch.cat([q_b, c_chunk], dim=1)  # [1, Q+Lc, H]

#                 # 若干层 self-attention 编码（用 StabilizedCrossAttention 当 self-attn）
#                 attn_w_last = None
#                 for layer in self.layers:
#                     out, attn_w, _ = layer["attn"](query=seq, key=seq, value=seq)
#                     seq = layer["norm"](seq + out)
#                     attn_w_last = attn_w

#                 # 编码后的 question / chunk 表示
#                 seq_enc = seq                            # [1,Q+Lc,H]
#                 q_enc = seq_enc[:, :Q, :]                # [1,Q,H]
#                 c_enc = seq_enc[:, Q:Q+Lc, :]            # [1,Lc,H]

#                 # 简单 pooling：Q / chunk 内做 mean
#                 q_repr = q_enc.mean(dim=1)               # [1,H]
#                 c_repr = c_enc.mean(dim=1)               # [1,H]

#                 # 注意力特征：最后一层的 question→chunk 注意力
#                 attn_features = []
#                 if self.use_attn_feature and attn_w_last is not None:
#                     # attn_w_last: [1,H,Q+Lc,Q+Lc]
#                     # 取前 Q 行（question token）指向后 Lc 列（chunk tokens）的注意力
#                     A_q2c = attn_w_last[:, :, :Q, Q:Q+Lc]    # [1,H,Q,Lc]

#                     # 一种比较稳的 pooling：
#                     # 1) 对 chunk tokens 做 max-pooling（强调最相关 token）
#                     # 2) 对 query 做 mean-pooling
#                     # 3) 对 heads 做 mean-pooling -> 得到一个标量 attention 特征
#                     # step1: max over chunk tokens -> [1,H,Q]
#                     w = A_q2c.max(dim=-1).values
#                     # step2: mean over Q -> [1,H]
#                     w = w.mean(dim=2)
#                     # step3: mean over heads -> [1]
#                     attn_scalar = w.mean(dim=1)          # [1]
#                     attn_features.append(attn_scalar)    # 列表里先放 [1]

#                 # 组装 MLP 输入特征
#                 feat_list = [q_repr, c_repr, q_repr * c_repr]   # 各 [1,H]
#                 if attn_features:
#                     # 把 [1] 扩成 [1,1] 再拼接
#                     attn_feat = attn_features[0].unsqueeze(-1)  # [1,1]
#                     feat_list.append(attn_feat)

#                 feat = torch.cat(feat_list, dim=-1)      # [1, 3H(+1)]
#                 logit = self.score_head(feat).squeeze(0).squeeze(-1)   # 标量 logit
#                 scores_b.append(logit)

#             scores_b = torch.stack(scores_b, dim=0)      # [J_b] (logits)

#             # 注意：这里默认返回 logits，normalize=False 直接用；如果你后面某些分析想看概率，
#             # 可以手动对 scores_b 做 softmax
#             if normalize and scores_b.numel() > 0:
#                 scores_b = torch.softmax(scores_b, dim=0)

#             scores_list.append(scores_b)

#         return scores_list





class LocalChunkScorer(nn.Module):
    """
    AttentionRAG 风格的局部检索（小 Transformer + 注意力特征 + MLP 打分）：
    - 对每个 chunk，构造 [question; chunk] 序列
    - 用若干层 StabilizedCrossAttention 当 self-attention 编码该序列
    - 从最后一层 self-attn 中取 question→chunk 的注意力，做 pooling 得到一个标量 attention 特征
    - 再用编码后的向量 + attention 特征，交给 MLP 输出 chunk logit（注意：这里输出的是 logits，不做 softmax）
    同时：
    - 可以额外返回每个 chunk 的“局部摘要特征”，供 GlobalChunkScorer 复用
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        num_layers: int = 2,
        use_attn_feature: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_attn_feature = use_attn_feature

        # 小型 Transformer encoder：堆几层 StabilizedCrossAttention，当 self-attn 用
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": StabilizedCrossAttention(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                ),
                "norm": nn.LayerNorm(hidden_size),
            })
            for _ in range(num_layers)
        ])

        # MLP 打分头：输入 = [q_repr, c_repr, q_repr⊙c_repr, attn_scalar]
        attn_feat_dim = 1 if use_attn_feature else 0
        in_dim = 3 * hidden_size + attn_feat_dim
        self.feature_dim = in_dim  # 暴露给全局评估器用

        self.score_head = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        question_emb: torch.Tensor,               # [B,Q,H]
        context_emb: torch.Tensor,                # [B,K,H]
        chunk_spans: List[List[Tuple[int, int]]],
        normalize: bool = False,                  # 默认返回 logits
        return_features: bool = False,            # 是否额外返回 chunk 特征
    ):
        """
        返回：
        - 如果 return_features=False:
            scores_list: List[Tensor[J_b]]，每个样本一条 chunk 分数向量（logits）
        - 如果 return_features=True:
            (scores_list, features_list)
            其中 features_list[b] 形状为 [J_b, feature_dim]
        """
        B, Q, H = question_emb.shape
        _, K, _ = context_emb.shape

        scores_list: List[torch.Tensor] = []
        features_list: List[torch.Tensor] = []  # 每个样本一个 [J_b, feature_dim]

        for b in range(B):
            spans_b = chunk_spans[b] if b < len(chunk_spans) else []
            if not spans_b:
                scores_list.append(context_emb.new_zeros(0))
                if return_features:
                    features_list.append(context_emb.new_zeros(0, self.feature_dim))
                continue

            q_b = question_emb[b:b+1]    # [1,Q,H]
            scores_b = []
            feats_b = []  # 这一条样本所有 chunk 的特征

            for (s, e) in spans_b:
                # 保守 clamp span
                s_clamped = max(0, min(K, s))
                e_clamped = max(s_clamped, min(K, e))
                if e_clamped <= s_clamped:
                    scores_b.append(context_emb.new_tensor(0.0))
                    if return_features:
                        feats_b.append(context_emb.new_zeros(self.feature_dim))
                    continue

                c_chunk = context_emb[b:b+1, s_clamped:e_clamped, :]   # [1,Lc,H]
                Lc = e_clamped - s_clamped

                # 1) [question; chunk] 序列
                seq = torch.cat([q_b, c_chunk], dim=1)  # [1, Q+Lc, H]

                # 2) 小 Transformer 编码
                attn_w_last = None
                for layer in self.layers:
                    out, attn_w, _ = layer["attn"](
                        query=seq,
                        key=seq,
                        value=seq,
                    )
                    seq = layer["norm"](seq + out)
                    attn_w_last = attn_w

                # 3) 编码后的 question / chunk 表示
                seq_enc = seq                            # [1,Q+Lc,H]
                q_enc = seq_enc[:, :Q, :]                # [1,Q,H]
                c_enc = seq_enc[:, Q:Q+Lc, :]            # [1,Lc,H]

                q_repr = q_enc.mean(dim=1)               # [1,H]
                c_repr = c_enc.mean(dim=1)               # [1,H]

                # 4) 注意力特征（question→chunk）
                attn_features = []
                if self.use_attn_feature and attn_w_last is not None:
                    # attn_w_last: [1,H,Q+Lc,Q+Lc]
                    A_q2c = attn_w_last[:, :, :Q, Q:Q+Lc]    # [1,H,Q,Lc]
                    w = A_q2c.max(dim=-1).values             # [1,H,Q]
                    w = w.mean(dim=2)                        # [1,H]
                    attn_scalar = w.mean(dim=1)              # [1]
                    attn_features.append(attn_scalar)

                # 5) 组装特征 + 打分
                feat_list = [q_repr, c_repr, q_repr * c_repr]   # 各 [1,H]
                if attn_features:
                    attn_feat = attn_features[0].unsqueeze(-1)  # [1,1]
                    feat_list.append(attn_feat)

                feat = torch.cat(feat_list, dim=-1)      # [1, feature_dim]
                logit = self.score_head(feat).squeeze(0).squeeze(-1)   # 标量 logit

                scores_b.append(logit)
                if return_features:
                    feats_b.append(feat.squeeze(0))      # [feature_dim]

            scores_b = torch.stack(scores_b, dim=0)      # [J_b]
            scores_list.append(
                torch.softmax(scores_b, dim=0) if (normalize and scores_b.numel() > 0) else scores_b
            )

            if return_features:
                if feats_b:
                    feats_b = torch.stack(feats_b, dim=0)    # [J_b, feature_dim]
                else:
                    feats_b = context_emb.new_zeros(0, self.feature_dim)
                features_list.append(feats_b)

        if return_features:
            return scores_list, features_list
        return scores_list



# class GlobalChunkScorer(nn.Module):
#     """
#     全局检索：
#     - 把所有 chunk 拼在一起，用一遍 cross-attn
#     - 通过 chunk_pooling_scores 在 chunk 级别做 pooling
#     """
#     def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
#         super().__init__()
#         self.cross_attn = StabilizedCrossAttention(
#             hidden_size=hidden_size,
#             num_heads=num_heads,
#             dropout=dropout,
#         )

#     def forward(
#         self,
#         question_emb: torch.Tensor,               # [B,Q,H]
#         context_emb: torch.Tensor,                # [B,K,H]
#         chunk_spans: List[List[Tuple[int, int]]],
#         normalize: bool = True,
#     ) -> List[torch.Tensor]:
#         """
#         返回：List[Tensor[J_b]]，每个样本一条 chunk 分数向量。
#         这里“全局视角”，所有 chunk 一起参与 cross-attn。
#         """
#         B, Q, H = question_emb.shape
#         _, K, _ = context_emb.shape

#         cross_out, cross_weights, _ = self.cross_attn(
#             query=question_emb,
#             key=context_emb,
#             value=context_emb,
#         )  # cross_weights: [B,H,Q,K]

#         scores = chunk_pooling_scores(
#             cross_weights=cross_weights,
#             chunk_spans=chunk_spans,
#             reduce_heads=True,
#             reduce_query=True,
#             normalize=normalize,
#         )
#         return scores

class GlobalChunkScorer(nn.Module):
    """
    Chunk-level 全局评估器：
    - 输入：来自 LocalChunkScorer 的 chunk 摘要特征 [J_b, D_local]
    - 在 chunk 序列上跑一个小型 self-attention encoder（长度 J 很小）
    - 输出：每个 chunk 的全局 logit
    """
    def __init__(
        self,
        local_feat_dim: int,
        global_hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_chunks: int = 32,
    ):
        super().__init__()
        self.local_feat_dim = local_feat_dim
        self.global_hidden_dim = global_hidden_dim

        # 1) 先把 Local 特征降到较小维度，节省参数
        self.proj = nn.Linear(local_feat_dim, global_hidden_dim)

        # 2) chunk 位置编码（因为 J 很小，用简单的 Embedding 即可）
        self.pos_emb = nn.Embedding(max_chunks, global_hidden_dim)

        # 3) 在 chunk 序列上堆几层小 self-attn（用 StabilizedCrossAttention）
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": StabilizedCrossAttention(
                    hidden_size=global_hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                ),
                "norm": nn.LayerNorm(global_hidden_dim),
            })
            for _ in range(num_layers)
        ])

        # 4) 全局打分头：每个 chunk 一个 logit
        self.score_head = nn.Sequential(
            nn.Linear(global_hidden_dim, global_hidden_dim),
            nn.GELU(),
            nn.Linear(global_hidden_dim, 1),
        )

    def forward(
        self,
        chunk_features_list: List[torch.Tensor],   # 每个样本: [J_b, local_feat_dim]
        normalize: bool = False,
    ) -> List[torch.Tensor]:
        """
        返回：
        - global_scores: List[Tensor[J_b]]，每个样本一条全局 logit 向量
        """
        global_scores: List[torch.Tensor] = []

        for feats_b in chunk_features_list:
            # feats_b: [J_b, local_feat_dim]
            if feats_b is None or feats_b.numel() == 0:
                global_scores.append(torch.zeros(0, device=self.proj.weight.device))
                continue

            J_b = feats_b.size(0)
            device = feats_b.device

            # 1) 映射到 global_hidden_dim
            h = self.proj(feats_b)    # [J_b, Dg]

            # 2) 加位置编码
            pos_ids = torch.arange(J_b, device=device)
            pos_ids = pos_ids.clamp(max=self.pos_emb.num_embeddings - 1)
            h = h + self.pos_emb(pos_ids)  # [J_b, Dg]

            # 3) 在 chunk 序列上做 self-attn
            seq = h.unsqueeze(0)  # [1, J_b, Dg]
            for layer in self.layers:
                out, _, _ = layer["attn"](
                    query=seq,
                    key=seq,
                    value=seq,
                )  # out: [1,J_b,Dg]
                seq = layer["norm"](seq + out)

            # 4) 每个 chunk 一个全局向量 + MLP 打分
            seq_enc = seq.squeeze(0)         # [J_b, Dg]
            logits_b = self.score_head(seq_enc).squeeze(-1)  # [J_b]

            if normalize and logits_b.numel() > 0:
                logits_b = torch.softmax(logits_b, dim=0)

            global_scores.append(logits_b)

        return global_scores



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


# class MultiViewChunkRetrieval(nn.Module):
#     """
#     评估器入口：
#     - LocalChunkScorer: 局部
#     - GlobalChunkScorer: 全局
#     - MultiViewEvidenceRouter: 融合 local/global
#     输出：
#     - final_scores: 用于剪枝 / loss
#     - local_scores, global_scores: 方便做 ablation / 辅助损失
#     - global_gate: 供后续模块使用（本版本模型暂未使用）
#     """
#     def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
#         super().__init__()
#         # self.local_scorer = DummyLocalChunkScorer(hidden_size)
#         self.local_scorer = LocalChunkScorer(hidden_size, num_heads, dropout)
#         # self.global_scorer = GlobalChunkScorer(hidden_size, num_heads, dropout)
#         # self.router = MultiViewEvidenceRouter(hidden_size)

#     def forward(
#         self,
#         question_emb: torch.Tensor,               # [B,Q,H]
#         context_emb: torch.Tensor,                # [B,K,H]
#         chunk_spans: List[List[Tuple[int,int]]],
#     ) -> Dict[str, object]:
#         local_scores = self.local_scorer(question_emb, context_emb, chunk_spans)
#         final_scores = local_scores
#         global_scores = [score.clone() for score in local_scores]  # 为了兼容后续接口
#         global_gate = None
#         # global_scores = self.global_scorer(question_emb, context_emb, chunk_spans)
#         # final_scores, global_gate = self.router(
#         #     question_emb=question_emb,
#         #     context_emb=context_emb,
#         #     local_scores=local_scores,
#         #     global_scores=global_scores,
#         #     chunk_spans=chunk_spans,
#         # )
#         return {
#             "final_scores": final_scores,
#             "local_scores": local_scores,
#             "global_scores": global_scores,
#             "global_gate": global_gate,
#         }


class MultiViewChunkRetrieval(nn.Module):
    """
    评估器入口：
    - LocalChunkScorer: 局部（token 级）
    - GlobalChunkScorer: 全局（chunk 级）
    - MultiViewEvidenceRouter: 融合 local/global
    输出：
    - final_scores: 用于剪枝 / loss
    - local_scores, global_scores: 方便做 ablation / 辅助损失
    - global_gate: 供后续模块使用（本版本模型暂未使用）
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1,enable_global:bool = True,):
        super().__init__()
        self.enable_global = enable_global
        # 1) 局部评估器（已经是小 Transformer + 注意力特征）
        self.local_scorer = LocalChunkScorer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=1,
            use_attn_feature=True,
        )


        if self.enable_global:
            # 2) 全局评估器：基于 local 的 chunk 特征
            #    这里选 global_hidden_dim=hidden_size//8 比较轻，你也可以调成 //4
            global_hidden_dim = hidden_size // 8
            global_heads = max(1, num_heads // 4)
            self.global_scorer = GlobalChunkScorer(
                local_feat_dim=self.local_scorer.feature_dim,
                global_hidden_dim=global_hidden_dim,
                num_heads=global_heads,
                num_layers=1,
                dropout=dropout,
                max_chunks=32,
            )

            # 3) 路由器：基于 question / chunk 语义做门控融合
            self.router = MultiViewEvidenceRouter(hidden_size)
        else :
            self.global_scorer = None
            self.router = None

    def forward(
        self,
        question_emb: torch.Tensor,               # [B,Q,H]
        context_emb: torch.Tensor,                # [B,K,H]
        chunk_spans: List[List[Tuple[int,int]]],
    ) -> Dict[str, object]:
        # 1) Local：scores + chunk-level features
        local_scores, chunk_features = self.local_scorer(
            question_emb,
            context_emb,
            chunk_spans,
            normalize=False,
            return_features=True,
        )   # local_scores: List[J_b] (logits)
            # chunk_features[b]: [J_b, D_local]


        if not self.enable_global:
            # Stage1：只用局部评估，final_scores = local_scores
            final_scores = local_scores
            # 为了接口兼容，global_scores 就 clone 一份，gate 设 None
            global_scores = [s.clone() for s in local_scores]
            global_gate = None
            return {
                "final_scores": final_scores,
                "local_scores": local_scores,
                "global_scores": global_scores,
                "global_gate": global_gate,
            }

        # 2) Stage2：local + global + router
        global_scores = self.global_scorer(
            chunk_features_list=chunk_features,
            normalize=False,)

        final_scores, global_gate = self.router(
            question_emb=question_emb,
            context_emb=context_emb,
            local_scores=local_scores,
            global_scores=global_scores,
            chunk_spans=chunk_spans,)

         

        return {
            "final_scores": final_scores,
            "local_scores": local_scores,
            "global_scores": global_scores,
            "global_gate": global_gate,
        }
