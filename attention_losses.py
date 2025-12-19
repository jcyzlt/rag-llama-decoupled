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


# class LatentContrastiveAlignmentLoss(nn.Module):
#     """
#     【隐层对比对齐损失 (LCA-Loss)】
#     Ref: InfoNCE Loss (Oord et al., 2018), SimCSE (Gao et al., 2021)
    
#     目标: 强迫融合模块生成的 Prefix 在隐层空间中：
#     1. 与 "仅由正确文档生成的前缀" (Positive) 靠近
#     2. 与 "由高分错误文档生成的前缀" (Negative) 远离
    
#     这能显著增强融合模块的抗噪性 (Noise Robustness)。
#     """
#     def __init__(self, temperature=0.07):
#         super().__init__()
#         self.temperature = temperature
#         self.cos_sim = nn.CosineSimilarity(dim=-1)
#         self.ce_loss = nn.CrossEntropyLoss()

#     def forward(self, anchor_emb, positive_emb, negative_emb):
#         """
#         Args:
#             anchor_emb:   [B, H] (当前训练的前缀，包含混合噪音)
#             positive_emb: [B, H] (纯净正确前缀)
#             negative_emb: [B, H] (错误诱导前缀)
#         """
#         # 1. 归一化 (Cosine Similarity 需要)
#         anchor_norm = F.normalize(anchor_emb, dim=1)
#         pos_norm = F.normalize(positive_emb, dim=1)
#         neg_norm = F.normalize(negative_emb, dim=1)

#         # 2. 计算相似度
#         # sim_pos: [B] -> [B, 1]
#         sim_pos = self.cos_sim(anchor_norm, pos_norm).unsqueeze(1)
#         # sim_neg: [B] -> [B, 1]
#         sim_neg = self.cos_sim(anchor_norm, neg_norm).unsqueeze(1)

#         # 3. 拼接 Logits [B, 2] (第0列是正例，第1列是负例)
#         logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature

#         # 4. 标签 (全是 0，因为第0列是正例)
#         labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

#         return self.ce_loss(logits, labels)
    
# attention_losses.py
class ZeroForwardLCASLoss(nn.Module):
    """
    零额外forward的LCA Loss：
    - 利用主路径cross-attn权重构造chunk表示
    - 支持in-batch negatives
    - 梯度自动隔离（通过detach）
    """
    def __init__(self, hidden_size, temperature_init=0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature_init))
        # ✅ 可学习的slot聚合器（替代粗暴mean）
        self.slot_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, prefix_slots, cross_weights, kv_embeddings, chunk_spans, chunk_labels):
        """
        prefix_slots: [B, M, H] - latent prefix表示
        cross_weights: [B, heads, M, K] - cross-attn权重
        kv_embeddings: [B, K, H] - 已detach的KV嵌入
        chunk_spans: List[List[(s,e)]] - chunk区间
        chunk_labels: List[Tensor[J]] - 0/1标签
        """
        B, M, H = prefix_slots.shape
        _, _, _, K = cross_weights.shape
        device = prefix_slots.device
        
        # 1. 对head维度平均 → [B, M, K]
        attn_weights = cross_weights.mean(dim=1)
        
        # 2. ✅ 可学习slot聚合得到anchor表示
        # 每个slot的重要性由gate网络动态决定
        slot_importance = self.slot_gate(prefix_slots)  # [B, M, 1]
        slot_importance = F.softmax(slot_importance, dim=1)
        anchors = (slot_importance * prefix_slots).sum(dim=1)  # [B, H]
        
        # 3. 构建batch内所有chunk的表示
        all_chunk_reps = []
        all_chunk_labels = []
        sample_offsets = [0]  # 记录每个样本的chunk起始索引
        
        for b in range(B):
            spans_b = chunk_spans[b]
            labels_b = chunk_labels[b].to(device)
            
            chunk_reps_b = []
            for j, (s, e) in enumerate(spans_b):
                s = max(0, min(K, s))
                e = max(s, min(K, e))
                if e <= s:
                    continue
                
                # ✅ 核心：attention-weighted pooling of KV tokens
                # [M, Lc] @ [Lc, H] → [M, H]
                chunk_attn = attn_weights[b, :, s:e]
                chunk_kv = kv_embeddings[b, s:e]
                slot_chunk_repr = torch.matmul(chunk_attn, chunk_kv)
                
                # slot聚合得到chunk向量
                chunk_repr = (slot_importance[b] * slot_chunk_repr).sum(dim=0)
                chunk_reps_b.append(chunk_repr)
            
            if chunk_reps_b:
                all_chunk_reps.append(torch.stack(chunk_reps_b))
                all_chunk_labels.append(labels_b)
                sample_offsets.append(sample_offsets[-1] + len(chunk_reps_b))
        
        if not all_chunk_reps:
            return torch.tensor(0.0, device=device)
        
        # 4. In-batch InfoNCE
        losses = []
        for b in range(B):
            anchor = anchors[b:b+1]  # [1, H]
            start = sample_offsets[b]
            end = sample_offsets[b+1]
            
            # 正例
            pos_mask = all_chunk_labels[b] == 1
            if pos_mask.sum() == 0:
                continue
            pos_reps = all_chunk_reps[b][pos_mask]  # [P, H]
            
            # 负例：本样本负例 + 其他样本所有chunks
            neg_reps = []
            # 本样本负例
            neg_mask = all_chunk_labels[b] == 0
            if neg_mask.sum() > 0:
                neg_reps.append(all_chunk_reps[b][neg_mask])
            # 其他样本chunks
            for other_b in range(B):
                if other_b == b:
                    continue
                if sample_offsets[other_b] < sample_offsets[other_b+1]:
                    neg_reps.append(all_chunk_reps[other_b])
            
            neg_reps = torch.cat(neg_reps, dim=0) if neg_reps else torch.empty(0, H, device=device)
            
            # InfoNCE
            pos_scores = F.cosine_similarity(anchor, pos_reps, dim=-1) / self.temperature
            neg_scores = F.cosine_similarity(anchor, neg_reps, dim=-1) / self.temperature
            
            loss_b = -torch.log(
                torch.exp(pos_scores).sum() / 
                (torch.exp(pos_scores).sum() + torch.exp(neg_scores).sum() + 1e-8)
            )
            losses.append(loss_b)
        
        return torch.stack(losses).mean()