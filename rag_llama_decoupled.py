# models/rag/rag_llama_decoupled.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

from transformers import AutoConfig, LlamaModel, LlamaPreTrainedModel

from double_attention import DoubleAttentionStack  
from multiview_chunk_retrieval import MultiViewChunkRetrieval

import torch.nn.functional as F

def compute_chunk_loss(
    chunk_scores_list: List[torch.Tensor],  # logits
    chunk_labels_list: Optional[List[torch.Tensor]],
    device: torch.device,
) -> torch.Tensor:
    if chunk_labels_list is None:
        return torch.tensor(0.0, device=device)

    losses = []
    for b_idx,(scores_b, labels_b) in enumerate(zip(chunk_scores_list, chunk_labels_list)):
        if scores_b is None or scores_b.numel() == 0:
            continue

        # 1) 检查 NaN/Inf
        if torch.isnan(scores_b).any() or torch.isinf(scores_b).any():
            print("[NaN DEBUG] found NaN/Inf in chunk_scores, sample_idx =", b_idx)
            print("  scores_b min/max:", scores_b.min().item(), scores_b.max().item())
            # 这里可以选择直接跳过这个样本，避免训练崩盘
            continue
            
        scores_b = scores_b.to(device).float()   # [J]
        labels_b = labels_b.to(device).long()    # [J]

        J = scores_b.size(0)
        if labels_b.numel() > J:
            labels_b = labels_b[:J]
        elif labels_b.numel() < J:
            pad = torch.zeros(J - labels_b.numel(), dtype=labels_b.dtype, device=device)
            labels_b = torch.cat([labels_b, pad], dim=0)

        pos_mask = labels_b == 1
        num_pos = pos_mask.sum()
        if num_pos == 0:
            continue

        # logits -> log_prob over chunks
        log_prob = F.log_softmax(scores_b, dim=0)  # [J]

        # 多正例：对所有正例 log p_pos 取平均
        loss_b = -log_prob[pos_mask].mean()
        losses.append(loss_b)

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()

# def compute_chunk_loss(
#     chunk_scores_list: List[torch.Tensor],
#     chunk_labels_list: Optional[List[torch.Tensor]],
#     device: torch.device,
# ) -> torch.Tensor:
#     """
#     简单版 chunk loss（方向 A 的逻辑）：
#     - chunk_scores_list: List[Tensor[J_b]]，已经 softmax 过
#     - chunk_labels_list: List[Tensor[J_b]]，0/1 标签
#     - 每条样本只有 1 个正样本时，退化为 -log p_pos
#     """
#     if chunk_labels_list is None:
#         return torch.tensor(0.0, device=device)

#     losses = []
#     for scores_b, labels_b in zip(chunk_scores_list, chunk_labels_list):
#         if scores_b is None or scores_b.numel() == 0:
#             continue
#         scores_b = scores_b.to(device).float()
#         labels_b = labels_b.to(device).long()

#         J = scores_b.size(0)
#         if labels_b.numel() > J:
#             labels_b = labels_b[:J]
#         elif labels_b.numel() < J:
#             pad = torch.zeros(J - labels_b.numel(), dtype=labels_b.dtype, device=device)
#             labels_b = torch.cat([labels_b, pad], dim=0)

#         pos_mask = labels_b == 1
#         if pos_mask.sum() == 0:
#             continue
#         probs_pos = scores_b[pos_mask]
#         loss_b = -torch.log(probs_pos + 1e-8).mean()
#         losses.append(loss_b)

#     if not losses:
#         return torch.tensor(0.0, device=device)

#     return torch.stack(losses).mean()


class RAGLlamaDecoupled(LlamaPreTrainedModel):
    """
    解耦版 RAG 模型：
    - 评估器：MultiViewChunkRetrieval（局部+全局）只用于 chunk 打分 + 剪枝
    - 融合器：DoubleAttentionStack 基于 pruned context 生成 prefix
    - LLaMA 主体：冻结，只用 [prefix, pruned_ctx_emb, answer_emb[:-1]] 做生成
    """
    def __init__(
        self,
        llama_model_name_or_path: str,
        num_da_layers: int = 1,
        num_heads: int = 16,
        dropout: float = 0.1,
        prompt_max_len: int = 16,
        chunk_loss_weight: float = 0.1,
        enable_global:bool = True,
    ):
        config = AutoConfig.from_pretrained(llama_model_name_or_path)
        super().__init__(config)
        self.enable_global = enable_global

        # 1) LLaMA 主体（冻结）
        self.llama = LlamaModel.from_pretrained(
            llama_model_name_or_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=None,
        )
        for p in self.llama.parameters():
            p.requires_grad = False

        self.hidden_size = config.hidden_size
        self.prompt_max_len = prompt_max_len
        self.chunk_loss_weight = chunk_loss_weight

        # 2) 融合器：DoubleAttentionStack（只吃 pruned context）
        self.double_stack = DoubleAttentionStack(
            hidden_dim=self.hidden_size,
            num_heads=num_heads,
            num_layers=num_da_layers,
            dropout=dropout,
        )

        # 3) 评估器：MultiViewChunkRetrieval（单独文件里的模块）
        self.retrieval = MultiViewChunkRetrieval(
            hidden_size=self.hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            enable_global = enable_global,
        )

        # 4) 输出头：tie 到 embedding
        self.lm_head = nn.Linear(self.hidden_size, config.vocab_size, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(self.llama.embed_tokens.weight)

        # 特殊 token id
        self.pad_id = config.pad_token_id if config.pad_token_id is not None else 0
        self.eos_id = config.eos_token_id if config.eos_token_id is not None else 2
        self.bos_id = config.bos_token_id if config.bos_token_id is not None else 1

        if self.pad_id == self.eos_id:
            raise ValueError("pad_token_id 必须与 eos_token_id 不同")

    def resize_output_embeddings(self, new_vocab_size: int):
        old_weight = self.lm_head.weight.data
        new_head = nn.Linear(old_weight.size(1), new_vocab_size, bias=False).to(old_weight.device)
        with torch.no_grad():
            copy = min(old_weight.size(0), new_vocab_size)
            new_head.weight[:copy] = old_weight[:copy]
        self.lm_head = new_head

    # ========== 上下文剪枝工具 ==========
    @staticmethod
    def _prune_context(
        context_ids: torch.Tensor,                  # [B,K_full]
        chunk_spans: List[List[Tuple[int,int]]],   # full context 下的 spans
        chunk_scores: List[torch.Tensor],          # [J_b]
        topk_chunks: int,
        device: torch.device,
    ):
        """
        根据 chunk_scores 剪枝，返回：
        - pruned_context_ids: [B,K_pruned_max]
        - pruned_chunk_spans: List[List[(s,e)]]（在剪枝后坐标系下）
        - ctx_mask: [B,K_pruned_max]（1=有效，0=pad）
        """
        B, K_full = context_ids.shape
        pruned_ids_list = []
        pruned_spans_list: List[List[Tuple[int,int]]] = []

        for b in range(B):
            spans_b = chunk_spans[b] if b < len(chunk_spans) else []
            scores_b = chunk_scores[b] if b < len(chunk_scores) else None

            # fallback：没有 spans 或 scores 时，保留整段 context
            if not spans_b or scores_b is None or scores_b.numel() == 0:
                ids_b = context_ids[b].tolist()
                pruned_ids_list.append(ids_b)
                pruned_spans_list.append([(0, len(ids_b))])
                continue

            J = len(spans_b)
            k_eff = min(topk_chunks, J)
            top_idx = torch.topk(scores_b, k=k_eff).indices.tolist()

            # 按得分排序（从大到小），以保持一定稳定性
            top_idx = sorted(top_idx, key=lambda i: float(scores_b[i]), reverse=True)

            new_ids_b: List[int] = []
            new_spans_b: List[Tuple[int,int]] = []
            cur = 0
            for j in top_idx:
                s, e = spans_b[j]
                s_clamped = max(0, min(K_full, s))
                e_clamped = max(s_clamped, min(K_full, e))
                if e_clamped <= s_clamped:
                    continue
                span_len = e_clamped - s_clamped
                new_spans_b.append((cur, cur + span_len))
                new_ids_b.extend(context_ids[b, s_clamped:e_clamped].tolist())
                cur += span_len

            if not new_ids_b:
                # 如果全是空 span，退回保留整段
                new_ids_b = context_ids[b].tolist()
                new_spans_b = [(0, len(new_ids_b))]

            pruned_ids_list.append(new_ids_b)
            pruned_spans_list.append(new_spans_b)

        # pad 成 batch tensor
        max_len = max(len(ids) for ids in pruned_ids_list) if pruned_ids_list else 0
        pruned_context_ids = context_ids.new_full((B, max_len), fill_value=0)
        ctx_mask = context_ids.new_zeros((B, max_len), dtype=torch.long)

        for b in range(B):
            ids_b = pruned_ids_list[b]
            Lb = len(ids_b)
            pruned_context_ids[b, :Lb] = torch.tensor(ids_b, device=device, dtype=context_ids.dtype)
            ctx_mask[b, :Lb] = 1

        return pruned_context_ids, pruned_spans_list, ctx_mask

    # ========== 前向 ==========
    def forward(
        self,
        question_ids: torch.Tensor,                 # [B,Q]
        context_ids: torch.Tensor,                  # [B,K_full]
        answer_ids: Optional[torch.Tensor] = None,  # [B,T]
        chunk_spans: Optional[List[List[Tuple[int,int]]]] = None,
        chunk_labels: Optional[List[torch.Tensor]] = None,
        topk_chunks: int = 2,
    ) -> Dict[str, torch.Tensor]:
        """
        训练:
          - question_ids, context_ids, answer_ids, chunk_spans, chunk_labels
          - 返回 total_loss / gen_loss / chunk_loss 等
        推理:
          - 不传 answer_ids（或 None），只返回 prefix / pruned_ctx_emb，方便外部调用 generate
        """
        device = question_ids.device
        B, Q = question_ids.shape
        _, K_full = context_ids.shape

        if chunk_spans is None:
            chunk_spans = [[] for _ in range(B)]

        # 1) Embedding（LLaMA 冻结）
        q_emb_full = self.llama.embed_tokens(question_ids)    # [B,Q,H]
        c_emb_full = self.llama.embed_tokens(context_ids)     # [B,K_full,H]

        # 2) 评估器：MultiViewChunkRetrieval（full context 上做局部+全局检索）
        retrieval_out = self.retrieval(q_emb_full, c_emb_full, chunk_spans)
        final_scores: List[torch.Tensor] = retrieval_out["final_scores"]
        local_scores = retrieval_out["local_scores"]
        global_scores = retrieval_out["global_scores"]
        global_gate = retrieval_out["global_gate"]  # 当前版本未在 LLaMA 中使用，后续可接 SSM

        # 3) 基于 final_scores 做剪枝，得到 pruned context + pruned spans
        pruned_context_ids, pruned_chunk_spans, ctx_mask = self._prune_context(
            context_ids=context_ids,
            chunk_spans=chunk_spans,
            chunk_scores=final_scores,
            topk_chunks=topk_chunks,
            device=device,
        )
        c_emb_pruned = self.llama.embed_tokens(pruned_context_ids)  # [B,K_pruned,H]

        # 4) 融合器：DoubleAttentionStack( question_emb, pruned_context_emb ) -> prefix
        prefix, attn_info = self.double_stack(q_emb_full, c_emb_pruned)  # [B,M,H]
        # 截断 prefix 长度
        if prefix.size(1) > self.prompt_max_len:
            prefix = prefix[:, -self.prompt_max_len:, :]
        B, M, H = prefix.shape

        # 推理场景：没有 answer_ids，只返回 prefix 和 pruned context，方便外部 generate
        if answer_ids is None:
            return {
                "prefix_emb": prefix,              # [B,M,H]
                "pruned_context_ids": pruned_context_ids,
                "pruned_context_emb": c_emb_pruned,
                "chunk_scores": final_scores,
                "local_scores": local_scores,
                "global_scores": global_scores,
                "global_gate": global_gate,
                "attention_info_fusion": attn_info,
            }

        # ======= 训练场景 =======

        # 5) 构造答案 Embedding（teacher forcing）
        # answer_ids: [B,T] -> 输入为 [:, :-1]，标签为 [:, 1:]
        ans_in = answer_ids[:, :-1]   # [B,T-1]
        ans_tgt = answer_ids[:, 1:]   # [B,T-1]
        ans_emb = self.llama.embed_tokens(ans_in)  # [B,T-1,H]
        Tm1 = ans_in.size(1)

        # 6) 拼接 inputs_embeds = [prefix, pruned_ctx_emb, ans_emb]
        inputs_embeds = torch.cat([prefix, c_emb_pruned, ans_emb], dim=1)  # [B, M+Kp+T-1, H]
        total_len = inputs_embeds.size(1)

        # attention_mask：全 1（也可以对 pad 的 ctx_mask 做 0，这里简单处理）
        prefix_mask = torch.ones(B, M, dtype=torch.long, device=device)
        attn_mask_ctx = ctx_mask  # [B,Kp]
        ans_mask = (ans_in != self.pad_id).long()
        attention_mask = torch.cat([prefix_mask, attn_mask_ctx, ans_mask], dim=1)  # [B,total_len]

        position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(B, -1)

        # 7) LLaMA 前向
        llama_outputs = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=False,
        )
        hidden_states = llama_outputs[0]  # [B,total_len,H]
        logits = self.lm_head(hidden_states)  # [B,total_len,V]

        # 8) 构造 labels：prefix + ctx 部分全 -100，答案部分为 ans_tgt
        labels_prefix = torch.full((B, M), -100, dtype=torch.long, device=device)
        labels_ctx = torch.full_like(attn_mask_ctx, -100)
        labels_ans = torch.where(
            ans_tgt != self.pad_id,
            ans_tgt,
            torch.full_like(ans_tgt, -100),
        )
        labels_for_ce = torch.cat([labels_prefix, labels_ctx, labels_ans], dim=1)  # [B,total_len]

        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_labels = labels_for_ce.reshape(-1)
        if (flat_labels != -100).sum() > 0:
            gen_loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
        else:
            gen_loss = logits.new_tensor(0.0)

        # 9) chunk_loss（来自 final_scores + chunk_labels）
        chunk_loss = compute_chunk_loss(final_scores, chunk_labels, device) * self.chunk_loss_weight

        total_loss = gen_loss + chunk_loss
        # total_loss = chunk_loss

        return {
            "logits": logits,
            "labels_for_ce": labels_for_ce,
            "total_loss": total_loss,
            "gen_loss": gen_loss,
            "chunk_loss": chunk_loss,
            "chunk_scores": final_scores,
            "local_scores": local_scores,
            "global_scores": global_scores,
            "global_gate": global_gate,
            "prefix_emb": prefix,
            "pruned_context_ids": pruned_context_ids,
            "pruned_context_emb": c_emb_pruned,
            "attention_info_fusion": attn_info,
        }
