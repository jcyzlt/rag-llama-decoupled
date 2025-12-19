# # models/rag/rag_llama_decoupled.py

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import List, Tuple, Dict, Optional

# from transformers import AutoConfig, LlamaModel, LlamaPreTrainedModel

# from double_attention import DoubleAttentionStack  
# from multiview_chunk_retrieval import MultiViewChunkRetrieval

# import torch.nn.functional as F
# from attention_losses import LatentContrastiveAlignmentLoss

# def compute_chunk_loss(
#     chunk_scores_list: List[torch.Tensor],  # logits
#     chunk_labels_list: Optional[List[torch.Tensor]],
#     device: torch.device,
# ) -> torch.Tensor:
#     if chunk_labels_list is None:
#         return torch.tensor(0.0, device=device)

#     losses = []
#     for b_idx,(scores_b, labels_b) in enumerate(zip(chunk_scores_list, chunk_labels_list)):
#         if scores_b is None or scores_b.numel() == 0:
#             continue

#         # 1) 检查 NaN/Inf
#         if torch.isnan(scores_b).any() or torch.isinf(scores_b).any():
#             print("[NaN DEBUG] found NaN/Inf in chunk_scores, sample_idx =", b_idx)
#             print("  scores_b min/max:", scores_b.min().item(), scores_b.max().item())
#             # 这里可以选择直接跳过这个样本，避免训练崩盘
#             continue
            
#         scores_b = scores_b.to(device).float()   # [J]
#         labels_b = labels_b.to(device).long()    # [J]

#         J = scores_b.size(0)
#         if labels_b.numel() > J:
#             labels_b = labels_b[:J]
#         elif labels_b.numel() < J:
#             pad = torch.zeros(J - labels_b.numel(), dtype=labels_b.dtype, device=device)
#             labels_b = torch.cat([labels_b, pad], dim=0)

#         pos_mask = labels_b == 1
#         num_pos = pos_mask.sum()
#         if num_pos == 0:
#             continue

#         # logits -> log_prob over chunks
#         log_prob = F.log_softmax(scores_b, dim=0)  # [J]

#         # 多正例：对所有正例 log p_pos 取平均
#         loss_b = -log_prob[pos_mask].mean()
#         losses.append(loss_b)

#     if not losses:
#         return torch.tensor(0.0, device=device)
#     return torch.stack(losses).mean()

# # def compute_chunk_loss(
# #     chunk_scores_list: List[torch.Tensor],
# #     chunk_labels_list: Optional[List[torch.Tensor]],
# #     device: torch.device,
# # ) -> torch.Tensor:
# #     """
# #     简单版 chunk loss（方向 A 的逻辑）：
# #     - chunk_scores_list: List[Tensor[J_b]]，已经 softmax 过
# #     - chunk_labels_list: List[Tensor[J_b]]，0/1 标签
# #     - 每条样本只有 1 个正样本时，退化为 -log p_pos
# #     """
# #     if chunk_labels_list is None:
# #         return torch.tensor(0.0, device=device)

# #     losses = []
# #     for scores_b, labels_b in zip(chunk_scores_list, chunk_labels_list):
# #         if scores_b is None or scores_b.numel() == 0:
# #             continue
# #         scores_b = scores_b.to(device).float()
# #         labels_b = labels_b.to(device).long()

# #         J = scores_b.size(0)
# #         if labels_b.numel() > J:
# #             labels_b = labels_b[:J]
# #         elif labels_b.numel() < J:
# #             pad = torch.zeros(J - labels_b.numel(), dtype=labels_b.dtype, device=device)
# #             labels_b = torch.cat([labels_b, pad], dim=0)

# #         pos_mask = labels_b == 1
# #         if pos_mask.sum() == 0:
# #             continue
# #         probs_pos = scores_b[pos_mask]
# #         loss_b = -torch.log(probs_pos + 1e-8).mean()
# #         losses.append(loss_b)

# #     if not losses:
# #         return torch.tensor(0.0, device=device)

# #     return torch.stack(losses).mean()


# class RAGLlamaDecoupled(LlamaPreTrainedModel):
#     """
#     解耦版 RAG 模型：
#     - 评估器：MultiViewChunkRetrieval（局部+全局）只用于 chunk 打分 + 剪枝
#     - 融合器：DoubleAttentionStack 基于 pruned context 生成 prefix
#     - LLaMA 主体：冻结，只用 [prefix, pruned_ctx_emb, answer_emb[:-1]] 做生成
#     """
#     def __init__(
#         self,
#         config,  # <--- 注意：参数名必须改为 config
#         num_da_layers: int = 1,
#         num_heads: int = 16,
#         dropout: float = 0.1,
#         prompt_max_len: int = 16,
#         chunk_loss_weight: float = 0.1,
#         lca_loss_weight: float = 0.2, 
#         enable_global: bool = True,
#         **kwargs
#     ):
#         # 兼容性逻辑：判断传入的是“模型路径字符串”还是“HuggingFace的配置对象”
#         if isinstance(config, str):
#             # 情况1：训练时传入的是路径字符串 (如 "./TinyLlama-1.1B")
#             # 我们需要手动加载 Config
#             llama_config = AutoConfig.from_pretrained(config)
#             super().__init__(llama_config)
#             # 加载基础模型权重
#             self.llama = LlamaModel.from_pretrained(config)
#         else:
#             # 情况2：推理时 from_pretrained 传入的是 Config 对象
#             llama_config = config
#             super().__init__(llama_config)
#             # 初始化空模型（权重随后会自动加载）
#             self.llama = LlamaModel(llama_config)
        

#         # 保存其他超参数
#         self.num_da_layers = num_da_layers
#         self.prompt_max_len = prompt_max_len
#         self.chunk_loss_weight = chunk_loss_weight
#         self.lca_loss_weight = lca_loss_weight
#         self.enable_global = enable_global

#         # 1) LLaMA 主体（冻结）
#         self.llama = LlamaModel.from_pretrained(
#             llama_config,
#             torch_dtype=torch.float16,
#             low_cpu_mem_usage=True,
#             device_map=None,
#         )
#         for p in self.llama.parameters():
#             p.requires_grad = False

#         self.hidden_size = config.hidden_size
#         self.prompt_max_len = prompt_max_len
#         self.chunk_loss_weight = chunk_loss_weight

#         # 2) 融合器：DoubleAttentionStack（只吃 pruned context）
#         self.double_stack = DoubleAttentionStack(
#             hidden_dim=self.hidden_size,
#             num_heads=num_heads,
#             num_layers=num_da_layers,
#             dropout=dropout,
#         )

#         # 3) 评估器：MultiViewChunkRetrieval（单独文件里的模块）
#         self.retrieval = MultiViewChunkRetrieval(
#             hidden_size=self.hidden_size,
#             num_heads=num_heads,
#             dropout=dropout,
#             enable_global = enable_global,
#         )

#         # 4) 输出头：tie 到 embedding
#         self.lm_head = nn.Linear(self.hidden_size, config.vocab_size, bias=False)
#         with torch.no_grad():
#             self.lm_head.weight.copy_(self.llama.embed_tokens.weight)

#         # 特殊 token id
#         self.pad_id = config.pad_token_id if config.pad_token_id is not None else 0
#         self.eos_id = config.eos_token_id if config.eos_token_id is not None else 2
#         self.bos_id = config.bos_token_id if config.bos_token_id is not None else 1

#         if self.pad_id == self.eos_id:
#             raise ValueError("pad_token_id 必须与 eos_token_id 不同")

#     def resize_output_embeddings(self, new_vocab_size: int):
#         old_weight = self.lm_head.weight.data
#         new_head = nn.Linear(old_weight.size(1), new_vocab_size, bias=False).to(old_weight.device)
#         with torch.no_grad():
#             copy = min(old_weight.size(0), new_vocab_size)
#             new_head.weight[:copy] = old_weight[:copy]
#         self.lm_head = new_head

#     # ========== 上下文剪枝工具 ==========
#     @staticmethod
#     def _prune_context(
#         context_ids: torch.Tensor,                  # [B,K_full]
#         chunk_spans: List[List[Tuple[int,int]]],   # full context 下的 spans
#         chunk_scores: List[torch.Tensor],          # [J_b]
#         topk_chunks: int,
#         device: torch.device,
#     ):
#         """
#         根据 chunk_scores 剪枝，返回：
#         - pruned_context_ids: [B,K_pruned_max]
#         - pruned_chunk_spans: List[List[(s,e)]]（在剪枝后坐标系下）
#         - ctx_mask: [B,K_pruned_max]（1=有效，0=pad）
#         """
#         B, K_full = context_ids.shape
#         pruned_ids_list = []
#         pruned_spans_list: List[List[Tuple[int,int]]] = []

#         for b in range(B):
#             spans_b = chunk_spans[b] if b < len(chunk_spans) else []
#             scores_b = chunk_scores[b] if b < len(chunk_scores) else None

#             # fallback：没有 spans 或 scores 时，保留整段 context
#             if not spans_b or scores_b is None or scores_b.numel() == 0:
#                 ids_b = context_ids[b].tolist()
#                 pruned_ids_list.append(ids_b)
#                 pruned_spans_list.append([(0, len(ids_b))])
#                 continue

#             J = len(spans_b)
#             k_eff = min(topk_chunks, J)
#             top_idx = torch.topk(scores_b, k=k_eff).indices.tolist()

#             # 按得分排序（从大到小），以保持一定稳定性
#             top_idx = sorted(top_idx, key=lambda i: float(scores_b[i]), reverse=True)

#             new_ids_b: List[int] = []
#             new_spans_b: List[Tuple[int,int]] = []
#             cur = 0
#             for j in top_idx:
#                 s, e = spans_b[j]
#                 s_clamped = max(0, min(K_full, s))
#                 e_clamped = max(s_clamped, min(K_full, e))
#                 if e_clamped <= s_clamped:
#                     continue
#                 span_len = e_clamped - s_clamped
#                 new_spans_b.append((cur, cur + span_len))
#                 new_ids_b.extend(context_ids[b, s_clamped:e_clamped].tolist())
#                 cur += span_len

#             if not new_ids_b:
#                 # 如果全是空 span，退回保留整段
#                 new_ids_b = context_ids[b].tolist()
#                 new_spans_b = [(0, len(new_ids_b))]

#             pruned_ids_list.append(new_ids_b)
#             pruned_spans_list.append(new_spans_b)

#         # pad 成 batch tensor
#         max_len = max(len(ids) for ids in pruned_ids_list) if pruned_ids_list else 0
#         pruned_context_ids = context_ids.new_full((B, max_len), fill_value=0)
#         ctx_mask = context_ids.new_zeros((B, max_len), dtype=torch.long)

#         for b in range(B):
#             ids_b = pruned_ids_list[b]
#             Lb = len(ids_b)
#             pruned_context_ids[b, :Lb] = torch.tensor(ids_b, device=device, dtype=context_ids.dtype)
#             ctx_mask[b, :Lb] = 1

#         return pruned_context_ids, pruned_spans_list, ctx_mask

#     # ========== 前向 ==========
#     def forward(
#         self,
#         question_ids: torch.Tensor,                 # [B,Q]
#         context_ids: torch.Tensor,                  # [B,K_full]
#         answer_ids: Optional[torch.Tensor] = None,  # [B,T]
#         chunk_spans: Optional[List[List[Tuple[int,int]]]] = None,
#         chunk_labels: Optional[List[torch.Tensor]] = None,
#         topk_chunks: int = 2,
#     ) -> Dict[str, torch.Tensor]:
#         """
#         训练:
#           - question_ids, context_ids, answer_ids, chunk_spans, chunk_labels
#           - 返回 total_loss / gen_loss / chunk_loss 等
#         推理:
#           - 不传 answer_ids（或 None），只返回 prefix / pruned_ctx_emb，方便外部调用 generate
#         """
#         device = question_ids.device
#         B, Q = question_ids.shape
#         _, K_full = context_ids.shape

#         if chunk_spans is None:
#             chunk_spans = [[] for _ in range(B)]

#         # 1) Embedding（LLaMA 冻结）
#         q_emb_full = self.llama.embed_tokens(question_ids)    # [B,Q,H]
#         c_emb_full = self.llama.embed_tokens(context_ids)     # [B,K_full,H]

#         # 2) 评估器：MultiViewChunkRetrieval（full context 上做局部+全局检索）
#         retrieval_out = self.retrieval(q_emb_full, c_emb_full, chunk_spans)
#         final_scores: List[torch.Tensor] = retrieval_out["final_scores"]
#         local_scores = retrieval_out["local_scores"]
#         global_scores = retrieval_out["global_scores"]
#         global_gate = retrieval_out["global_gate"]  # 当前版本未在 LLaMA 中使用，后续可接 SSM

#         # 3) 基于 final_scores 做剪枝，得到 pruned context + pruned spans
#         pruned_context_ids, pruned_chunk_spans, ctx_mask = self._prune_context(
#             context_ids=context_ids,
#             chunk_spans=chunk_spans,
#             chunk_scores=final_scores,
#             topk_chunks=topk_chunks,
#             device=device,
#         )
#         c_emb_pruned = self.llama.embed_tokens(pruned_context_ids)  # [B,K_pruned,H]

#         # 4) 融合器：DoubleAttentionStack( question_emb, pruned_context_emb ) -> prefix
#         prefix, attn_info = self.double_stack(q_emb_full, c_emb_pruned)  # [B,M,H]
#         # 截断 prefix 长度
#         if prefix.size(1) > self.prompt_max_len:
#             prefix = prefix[:, -self.prompt_max_len:, :]
#         B, M, H = prefix.shape

#         # 推理场景：没有 answer_ids，只返回 prefix 和 pruned context，方便外部 generate
#         if answer_ids is None:
#             return {
#                 "prefix_emb": prefix,              # [B,M,H]
#                 "pruned_context_ids": pruned_context_ids,
#                 "pruned_context_emb": c_emb_pruned,
#                 "chunk_scores": final_scores,
#                 "local_scores": local_scores,
#                 "global_scores": global_scores,
#                 "global_gate": global_gate,
#                 "attention_info_fusion": attn_info,
#             }

#         # ======= 训练场景 =======

#         # 5) 构造答案 Embedding（teacher forcing）
#         # answer_ids: [B,T] -> 输入为 [:, :-1]，标签为 [:, 1:]
#         ans_in = answer_ids[:, :-1]   # [B,T-1]
#         ans_tgt = answer_ids[:, 1:]   # [B,T-1]
#         ans_emb = self.llama.embed_tokens(ans_in)  # [B,T-1,H]
#         Tm1 = ans_in.size(1)

#         # 6) 拼接 inputs_embeds = [prefix, pruned_ctx_emb, ans_emb]
#         inputs_embeds = torch.cat([prefix, c_emb_pruned, ans_emb], dim=1)  # [B, M+Kp+T-1, H]
#         total_len = inputs_embeds.size(1)

#         # attention_mask：全 1（也可以对 pad 的 ctx_mask 做 0，这里简单处理）
#         prefix_mask = torch.ones(B, M, dtype=torch.long, device=device)
#         attn_mask_ctx = ctx_mask  # [B,Kp]
#         ans_mask = (ans_in != self.pad_id).long()
#         attention_mask = torch.cat([prefix_mask, attn_mask_ctx, ans_mask], dim=1)  # [B,total_len]

#         position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(B, -1)

#         # 7) LLaMA 前向
#         llama_outputs = self.llama(
#             inputs_embeds=inputs_embeds,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             return_dict=False,
#         )
#         hidden_states = llama_outputs[0]  # [B,total_len,H]
#         logits = self.lm_head(hidden_states)  # [B,total_len,V]

#         # 8) 构造 labels：prefix + ctx 部分全 -100，答案部分为 ans_tgt
#         labels_prefix = torch.full((B, M), -100, dtype=torch.long, device=device)
#         labels_ctx = torch.full_like(attn_mask_ctx, -100)
#         labels_ans = torch.where(
#             ans_tgt != self.pad_id,
#             ans_tgt,
#             torch.full_like(ans_tgt, -100),
#         )
#         labels_for_ce = torch.cat([labels_prefix, labels_ctx, labels_ans], dim=1)  # [B,total_len]

#         flat_logits = logits.reshape(-1, logits.size(-1))
#         flat_labels = labels_for_ce.reshape(-1)
#         if (flat_labels != -100).sum() > 0:
#             gen_loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
#         else:
#             gen_loss = logits.new_tensor(0.0)

#         # 9) chunk_loss（来自 final_scores + chunk_labels）
#         chunk_loss = compute_chunk_loss(final_scores, chunk_labels, device) * self.chunk_loss_weight

#         total_loss = gen_loss + chunk_loss
#         # total_loss = chunk_loss

#         return {
#             "logits": logits,
#             "labels_for_ce": labels_for_ce,
#             "total_loss": total_loss,
#             "gen_loss": gen_loss,
#             "chunk_loss": chunk_loss,
#             "chunk_scores": final_scores,
#             "local_scores": local_scores,
#             "global_scores": global_scores,
#             "global_gate": global_gate,
#             "prefix_emb": prefix,
#             "pruned_context_ids": pruned_context_ids,
#             "pruned_context_emb": c_emb_pruned,
#             "attention_info_fusion": attn_info,
#         }




# models/rag/rag_llama_decoupled.py

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import List, Tuple, Dict, Optional

# from transformers import AutoConfig, LlamaModel, LlamaPreTrainedModel

# # 确保这些模块在你的 PYTHONPATH 下能找到，或者在同级目录
# from double_attention import DoubleAttentionStack  
# from multiview_chunk_retrieval import MultiViewChunkRetrieval
# from attention_losses import LatentContrastiveAlignmentLoss

# def compute_chunk_loss(
#     chunk_scores_list: List[torch.Tensor],  # logits
#     chunk_labels_list: Optional[List[torch.Tensor]],
#     device: torch.device,
# ) -> torch.Tensor:
#     if chunk_labels_list is None:
#         return torch.tensor(0.0, device=device)

#     losses = []
#     for b_idx, (scores_b, labels_b) in enumerate(zip(chunk_scores_list, chunk_labels_list)):
#         if scores_b is None or scores_b.numel() == 0:
#             continue

#         if torch.isnan(scores_b).any() or torch.isinf(scores_b).any():
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
#         num_pos = pos_mask.sum()
#         if num_pos == 0:
#             continue

#         log_prob = F.log_softmax(scores_b, dim=0)
#         loss_b = -log_prob[pos_mask].mean()
#         losses.append(loss_b)

#     if not losses:
#         return torch.tensor(0.0, device=device)
#     return torch.stack(losses).mean()


# class RAGLlamaDecoupled(LlamaPreTrainedModel):
#     def __init__(
#         self,
#         config,  # <--- 修改点：参数名必须是 config
#         num_da_layers: int = 1,
#         num_heads: int = 16,
#         dropout: float = 0.1,
#         prompt_max_len: int = 16,
#         chunk_loss_weight: float = 1.0,
#         lca_loss_weight: float = 0.2, 
#         enable_global: bool = True,
#         **kwargs
#     ):
#         # --- 核心修改逻辑开始 ---
#         # 兼容性判断：处理 HuggingFace 自动传入的 Config 对象 或 训练时传入的 路径字符串
#         if isinstance(config, str):
#             # Case 1: 训练脚本传入了路径 (str)
#             llama_config = AutoConfig.from_pretrained(config)
#             super().__init__(llama_config)
#             # 加载基础模型权重
#             self.llama = LlamaModel.from_pretrained(config)
#         else:
#             # Case 2: 推理脚本传入了 Config 对象
#             llama_config = config
#             super().__init__(llama_config)
#             # 初始化空模型（HuggingFace 会在 init 结束后自动加载微调权重）
#             self.llama = LlamaModel(llama_config)
#         # --- 核心修改逻辑结束 ---

#         self.hidden_size = llama_config.hidden_size
#         self.num_da_layers = num_da_layers
#         self.prompt_max_len = prompt_max_len
#         self.chunk_loss_weight = chunk_loss_weight
#         self.lca_loss_weight = lca_loss_weight
#         self.enable_global = enable_global

#         # 如果是推理模式加载的空模型，为了防止生成时 device 错误，可以先不冻结，
#         # 等权重加载完再处理。但这里为了保持一致性，我们依然设置 False。
#         for p in self.llama.parameters():
#             p.requires_grad = False

#         self.double_stack = DoubleAttentionStack(
#             hidden_dim=self.hidden_size,
#             num_heads=num_heads,
#             num_layers=num_da_layers,
#             dropout=dropout,
#         )

#         self.retrieval = MultiViewChunkRetrieval(
#             hidden_size=self.hidden_size,
#             num_heads=num_heads,
#             dropout=dropout,
#             enable_global = enable_global,
#         )

#         self.lm_head = nn.Linear(self.hidden_size, llama_config.vocab_size, bias=False)
#         # 注意：如果是推理加载，这里会被微调权重覆盖，所以不用担心 copy 基础权重的问题
        
#         self.pad_id = llama_config.pad_token_id if llama_config.pad_token_id is not None else 0
#         self.eos_id = llama_config.eos_token_id if llama_config.eos_token_id is not None else 2
        
#         if hasattr(llama_config, "bos_token_id"):
#              self.bos_id = llama_config.bos_token_id
#         else:
#              self.bos_id = 1

#     def resize_output_embeddings(self, new_vocab_size: int):
#         old_weight = self.lm_head.weight.data
#         new_head = nn.Linear(old_weight.size(1), new_vocab_size, bias=False).to(old_weight.device)
#         with torch.no_grad():
#             copy = min(old_weight.size(0), new_vocab_size)
#             new_head.weight[:copy] = old_weight[:copy]
#         self.lm_head = new_head

#     @staticmethod
#     def _prune_context(
#         context_ids: torch.Tensor,
#         chunk_spans: List[List[Tuple[int,int]]],
#         chunk_scores: List[torch.Tensor],
#         topk_chunks: int,
#         device: torch.device,
#     ):
#         B, K_full = context_ids.shape
#         pruned_ids_list = []
#         pruned_spans_list: List[List[Tuple[int,int]]] = []

#         for b in range(B):
#             spans_b = chunk_spans[b] if b < len(chunk_spans) else []
#             scores_b = chunk_scores[b] if b < len(chunk_scores) else None

#             if not spans_b or scores_b is None or scores_b.numel() == 0:
#                 ids_b = context_ids[b].tolist()
#                 pruned_ids_list.append(ids_b)
#                 pruned_spans_list.append([(0, len(ids_b))])
#                 continue

#             J = len(spans_b)
#             k_eff = min(topk_chunks, J)
#             top_idx = torch.topk(scores_b, k=k_eff).indices.tolist()
#             top_idx = sorted(top_idx, key=lambda i: float(scores_b[i]), reverse=True)

#             new_ids_b: List[int] = []
#             new_spans_b: List[Tuple[int,int]] = []
#             cur = 0
#             for j in top_idx:
#                 s, e = spans_b[j]
#                 s_clamped = max(0, min(K_full, s))
#                 e_clamped = max(s_clamped, min(K_full, e))
#                 if e_clamped <= s_clamped:
#                     continue
#                 span_len = e_clamped - s_clamped
#                 new_spans_b.append((cur, cur + span_len))
#                 new_ids_b.extend(context_ids[b, s_clamped:e_clamped].tolist())
#                 cur += span_len

#             if not new_ids_b:
#                 new_ids_b = context_ids[b].tolist()
#                 new_spans_b = [(0, len(new_ids_b))]

#             pruned_ids_list.append(new_ids_b)
#             pruned_spans_list.append(new_spans_b)

#         max_len = max(len(ids) for ids in pruned_ids_list) if pruned_ids_list else 0
#         pruned_context_ids = context_ids.new_full((B, max_len), fill_value=0)
#         ctx_mask = context_ids.new_zeros((B, max_len), dtype=torch.long)

#         for b in range(B):
#             ids_b = pruned_ids_list[b]
#             Lb = len(ids_b)
#             pruned_context_ids[b, :Lb] = torch.tensor(ids_b, device=device, dtype=context_ids.dtype)
#             ctx_mask[b, :Lb] = 1

#         return pruned_context_ids, pruned_spans_list, ctx_mask

#     def forward(
#         self,
#         question_ids: torch.Tensor,
#         context_ids: torch.Tensor,
#         answer_ids: Optional[torch.Tensor] = None,
#         chunk_spans: Optional[List[List[Tuple[int,int]]]] = None,
#         chunk_labels: Optional[List[torch.Tensor]] = None,
#         topk_chunks: int = 2,
#     ) -> Dict[str, torch.Tensor]:
#         device = question_ids.device
#         B, Q = question_ids.shape

#         if chunk_spans is None:
#             chunk_spans = [[] for _ in range(B)]

#         q_emb_full = self.llama.embed_tokens(question_ids)
#         c_emb_full = self.llama.embed_tokens(context_ids)

#         retrieval_out = self.retrieval(q_emb_full, c_emb_full, chunk_spans)
#         final_scores: List[torch.Tensor] = retrieval_out["final_scores"]
#         local_scores = retrieval_out["local_scores"]
#         global_scores = retrieval_out["global_scores"]
#         global_gate = retrieval_out["global_gate"]

#         pruned_context_ids, pruned_chunk_spans, ctx_mask = self._prune_context(
#             context_ids=context_ids,
#             chunk_spans=chunk_spans,
#             chunk_scores=final_scores,
#             topk_chunks=topk_chunks,
#             device=device,
#         )
#         c_emb_pruned = self.llama.embed_tokens(pruned_context_ids)

#         prefix, attn_info = self.double_stack(q_emb_full, c_emb_pruned)
#         if prefix.size(1) > self.prompt_max_len:
#             prefix = prefix[:, -self.prompt_max_len:, :]
#         B, M, H = prefix.shape

#         # 推理模式出口
#         if answer_ids is None:
#             return {
#                 "prefix_emb": prefix,
#                 "pruned_context_ids": pruned_context_ids,
#                 "pruned_context_emb": c_emb_pruned,
#                 "chunk_scores": final_scores,
#                 "question_emb": q_emb_full,
#                 "local_scores": local_scores,
#                 "global_scores": global_scores,
#                 "attention_info_fusion": attn_info,
#             }

#         # 训练模式逻辑
#         ans_in = answer_ids[:, :-1]
#         ans_tgt = answer_ids[:, 1:]
#         ans_emb = self.llama.embed_tokens(ans_in)

#         inputs_embeds = torch.cat([prefix, c_emb_pruned,q_emb_full, ans_emb], dim=1)
#         total_len = inputs_embeds.size(1)

#         prefix_mask = torch.ones(B, M, dtype=torch.long, device=device)
#         attn_mask_ctx = ctx_mask
#         q_mask = torch.ones(B, Q, dtype=torch.long, device=device) # <--- 新增 Query Mask
#         ans_mask = (ans_in != self.pad_id).long()

#         attention_mask = torch.cat([prefix_mask, attn_mask_ctx,q_mask, ans_mask], dim=1)
        
#         position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(B, -1)

#         llama_outputs = self.llama(
#             inputs_embeds=inputs_embeds,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             return_dict=False,
#         )
#         hidden_states = llama_outputs[0]
#         logits = self.lm_head(hidden_states)

#         labels_prefix = torch.full((B, M), -100, dtype=torch.long, device=device)
#         labels_ctx = torch.full_like(attn_mask_ctx, -100)
#         labels_q = torch.full((B, Q), -100, dtype=torch.long, device=device) # <--- 新增 Query Label
#         labels_ans = torch.where(
#             ans_tgt != self.pad_id,
#             ans_tgt,
#             torch.full_like(ans_tgt, -100),
#         )
#         labels_for_ce = torch.cat([labels_prefix, labels_ctx,labels_q, labels_ans], dim=1)

#         flat_logits = logits.reshape(-1, logits.size(-1))
#         flat_labels = labels_for_ce.reshape(-1)
#         if (flat_labels != -100).sum() > 0:
#             gen_loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
#         else:
#             gen_loss = logits.new_tensor(0.0)

#         chunk_loss = compute_chunk_loss(final_scores, chunk_labels, device) * self.chunk_loss_weight
#         total_loss = gen_loss + chunk_loss

#         return {
#             "logits": logits,
#             "total_loss": total_loss,
#             "gen_loss": gen_loss,
#             "chunk_loss": chunk_loss,
#             "chunk_scores": final_scores,
#         }


# rag_llama_decoupled.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from transformers import AutoConfig, LlamaModel, LlamaPreTrainedModel

from double_attention import DoubleAttentionStack  
from multiview_chunk_retrieval import MultiViewChunkRetrieval
from attention_losses import ZeroForwardLCASLoss  # 确保你已有这个文件
from lexical_projection import MemoryEfficientTopKProjector

# def compute_chunk_loss(
#     chunk_scores_list: List[torch.Tensor],
#     chunk_labels_list: Optional[List[torch.Tensor]],
#     device: torch.device,
# ) -> torch.Tensor:
#     if chunk_labels_list is None:
#         return torch.tensor(0.0, device=device)

#     losses = []
#     for scores_b, labels_b in zip(chunk_scores_list, chunk_labels_list):
#         if scores_b is None or scores_b.numel() == 0: continue
#         if torch.isnan(scores_b).any() or torch.isinf(scores_b).any(): continue
            
#         scores_b = scores_b.to(device).float()
#         labels_b = labels_b.to(device).long()
#         J = scores_b.size(0)
        
#         if labels_b.numel() > J: labels_b = labels_b[:J]
#         elif labels_b.numel() < J:
#             pad = torch.zeros(J - labels_b.numel(), dtype=labels_b.dtype, device=device)
#             labels_b = torch.cat([labels_b, pad], dim=0)

#         pos_mask = labels_b == 1
#         if pos_mask.sum() == 0: continue

#         log_prob = F.log_softmax(scores_b, dim=0)
#         losses.append(-log_prob[pos_mask].mean())

#     if not losses: return torch.tensor(0.0, device=device)
#     return torch.stack(losses).mean()

# def compute_chunk_loss(
#     self,
#     chunk_scores,
#     chunk_labels,
#     margin: float = 0.2,
#     ce_weight: float = 0.5,
    
# ):
#     """
#     chunk_scores: List[Tensor[J_b]]，每个样本的一维分数
#     chunk_labels: List[Tensor[J_b]]，0/1
#     返回: scalar loss
#     """

#     ce_losses = []
#     rank_losses = []

#     for scores_b, labels_b in zip(chunk_scores, chunk_labels):
#         if scores_b is None or scores_b.numel() == 0:
#             continue

#         scores_b = scores_b.view(-1)               # [J_b]
#         labels_b = labels_b.to(scores_b.device).view(-1)

#         pos_mask = labels_b == 1
#         neg_mask = labels_b == 0

#         num_pos = pos_mask.sum().item()
#         num_neg = neg_mask.sum().item()

#         # 如果没有正例或没有负例，这个样本对 ranking 没用，直接跳过
#         if num_pos == 0 or num_neg == 0:
#             continue

#         # ====== 1) 原来的 CE-style loss（可选） ======
#         if ce_weight > 0:
#             log_prob = F.log_softmax(scores_b, dim=0)
#             ce_loss_b = -log_prob[pos_mask].mean()
#             ce_losses.append(ce_loss_b)

#         # ====== 2) margin-based ranking loss ======
#         pos_scores = scores_b[pos_mask]            # [P]
#         neg_scores = scores_b[neg_mask]            # [N]

#         # 最难负例：分数最高的那个
#         hard_neg_score = neg_scores.max()          # scalar

#         # 对每个正例都和同一个 hard_neg 比较
#         # loss_i = max(0, margin - s_pos + s_neg_hard)
#         rank_loss_b = F.relu(margin - pos_scores + hard_neg_score).mean()
#         rank_losses.append(rank_loss_b)

#     if not rank_losses and not ce_losses:
#         # 没有有效样本
#         return scores_b.new_tensor(0.0)

#     rank_loss = torch.stack(rank_losses).mean() if rank_losses else scores_b.new_tensor(0.0)
#     if ce_weight > 0 and ce_losses:
#         ce_loss = torch.stack(ce_losses).mean()
#         total = ce_weight * ce_loss + (1.0 - ce_weight) * rank_loss
#     else:
#         total = rank_loss

#     return total


def compute_chunk_loss(
    self,
    chunk_scores,
    chunk_labels,
    margin: float = 0.2,      # 保留以兼容旧调用；listwise里不再使用
    ce_weight: float = 0.0,   # 你原来传0：那就是纯 listwise
    temperature: float = 1.0, # 温度：0.5~2.0都可试，越小越“硬”
):
    """
    chunk_scores: List[Tensor[J_b]]，每个样本的一维分数(logits)
    chunk_labels: List[Tensor[J_b]]，0/1 (可多正例)
    返回: scalar loss

    Listwise 多正例 InfoNCE:
        L = -log( sum_{pos} exp(s_pos/tau) / sum_{all} exp(s_all/tau) )
          = -(logsumexp(pos) - logsumexp(all))
    可选混合旧的 CE-style（对正例logprob取mean）：
        ce_weight * ce_loss + (1-ce_weight) * listwise_loss
    """
    ce_losses = []
    list_losses = []

    # 用于返回0 loss时的 device/dtype 对齐
    fallback_tensor = None

    for scores_b, labels_b in zip(chunk_scores, chunk_labels):
        if scores_b is None or labels_b is None:
            continue

        # 可能是 Python list / CPU tensor，统一到 scores 的 device
        if not torch.is_tensor(scores_b):
            scores_b = torch.tensor(scores_b)

        if not torch.is_tensor(labels_b):
            labels_b = torch.tensor(labels_b)

        scores_b = scores_b.float()
        labels_b = labels_b.to(scores_b.device)

        fallback_tensor = scores_b  # 记录device

        if scores_b.numel() == 0:
            continue

        pos_mask = (labels_b > 0)
        num_pos = int(pos_mask.sum().item())
        if num_pos == 0:
            # 没正例样本：对 listwise 无监督信号，跳过
            continue

        # ====== Listwise multi-positive InfoNCE ======
        s = scores_b / max(temperature, 1e-6)
        lse_all = torch.logsumexp(s, dim=0)              # scalar
        lse_pos = torch.logsumexp(s[pos_mask], dim=0)    # scalar
        list_loss_b = -(lse_pos - lse_all)
        list_losses.append(list_loss_b)

        # ====== 可选：保留原 CE-style（正例 logprob mean） ======
        if ce_weight > 0:
            log_prob = F.log_softmax(s, dim=0)
            ce_loss_b = -log_prob[pos_mask].mean()
            ce_losses.append(ce_loss_b)

    if (not list_losses) and (not ce_losses):
        # 没有任何有效样本
        if fallback_tensor is None:
            return torch.tensor(0.0)
        return fallback_tensor.new_tensor(0.0)

    list_loss = torch.stack(list_losses).mean() if list_losses else fallback_tensor.new_tensor(0.0)

    if ce_weight > 0 and ce_losses:
        ce_loss = torch.stack(ce_losses).mean()
        total = ce_weight * ce_loss + (1.0 - ce_weight) * list_loss
    else:
        total = list_loss

    return total


class RAGLlamaDecoupled(LlamaPreTrainedModel):
    def __init__(
        self,
        config,
        num_da_layers: int = 2,
        num_heads: int =8,
        dropout: float = 0.1,
        prompt_max_len: int = 32,
        chunk_loss_weight: float = 1.0,
        lca_loss_weight: float = 0.2, 
        enable_global: bool = True,
        **kwargs
    ):
        if isinstance(config, str):
            llama_config = AutoConfig.from_pretrained(config)
            super().__init__(llama_config)
            self.llama = LlamaModel.from_pretrained(config)
        else:
            llama_config = config
            super().__init__(llama_config)
            self.llama = LlamaModel(llama_config)

        self.hidden_size = llama_config.hidden_size
        self.num_da_layers = num_da_layers
        self.prompt_max_len = prompt_max_len
        self.chunk_loss_weight = chunk_loss_weight
        self.lca_loss_weight = lca_loss_weight
        self.enable_global = enable_global
        self.num_heads = num_heads
        self.dropout = dropout

        # 冻结 LLaMA
        for p in self.llama.parameters():
            p.requires_grad = False

        self.double_stack = DoubleAttentionStack(
            hidden_dim=self.hidden_size,
            num_heads=num_heads,
            num_layers=num_da_layers,
            dropout=dropout,
        )

        self.retrieval = MultiViewChunkRetrieval(
            hidden_size=self.hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            enable_global = enable_global,
        )

        self.lex_proj = MemoryEfficientTopKProjector(
            hidden_size=self.hidden_size,
            vocab_size=self.llama.config.vocab_size,
            k=32,
            block_size=1024,  # 可根据显存调整
            tau_init=1.0
        )
        # 初始化 LCA Loss
        self.lca_loss_fct = ZeroForwardLCASLoss()

        self.lm_head = nn.Linear(self.hidden_size, llama_config.vocab_size, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(self.llama.embed_tokens.weight)
        self.pad_id = llama_config.pad_token_id or 0
        self.eos_id = llama_config.eos_token_id or 2

    def resize_output_embeddings(self, new_vocab_size: int):
        old_weight = self.lm_head.weight.data
        new_head = nn.Linear(old_weight.size(1), new_vocab_size, bias=False).to(old_weight.device)
        with torch.no_grad():
            copy = min(old_weight.size(0), new_vocab_size)
            new_head.weight[:copy] = old_weight[:copy]
        self.lm_head = new_head

    def _get_fused_rep(self, q_emb, specific_c_emb):
        """ LCA Helper: 只运行融合模块获取 Prefix """
        prefix, _ = self.double_stack(q_emb, specific_c_emb)
        return prefix.mean(dim=1)

    # @staticmethod
    # def _prune_context(context_ids, chunk_spans, chunk_scores, topk_chunks, device):
    #     B, K_full = context_ids.shape
    #     pruned_ids_list, pruned_spans_list = [], []

    #     for b in range(B):
    #         spans_b = chunk_spans[b] if b < len(chunk_spans) else []
    #         scores_b = chunk_scores[b] if b < len(chunk_scores) else None

    #         if not spans_b or scores_b is None or scores_b.numel() == 0:
    #             ids_b = context_ids[b].tolist()
    #             pruned_ids_list.append(ids_b)
    #             pruned_spans_list.append([(0, len(ids_b))])
    #             continue

    #         J = len(spans_b)
    #         k_eff = min(topk_chunks, J)
    #         # 降序排列
    #         top_idx = torch.topk(scores_b, k=k_eff).indices.tolist()
    #         top_idx = sorted(top_idx, key=lambda i: float(scores_b[i]), reverse=True)

    #         new_ids_b, new_spans_b = [], []
    #         cur = 0
    #         for j in top_idx:
    #             s, e = spans_b[j]
    #             s_clamped = max(0, min(K_full, s))
    #             e_clamped = max(s_clamped, min(K_full, e))
    #             if e_clamped <= s_clamped: continue
    #             span_len = e_clamped - s_clamped
    #             new_spans_b.append((cur, cur + span_len))
    #             new_ids_b.extend(context_ids[b, s_clamped:e_clamped].tolist())
    #             cur += span_len

    #         if not new_ids_b:
    #             new_ids_b = context_ids[b].tolist()
    #             new_spans_b = [(0, len(new_ids_b))]

    #         pruned_ids_list.append(new_ids_b)
    #         pruned_spans_list.append(new_spans_b)

    #     max_len = max(len(ids) for ids in pruned_ids_list) if pruned_ids_list else 0
    #     pruned_context_ids = context_ids.new_full((B, max_len), fill_value=0)
    #     ctx_mask = context_ids.new_zeros((B, max_len), dtype=torch.long)

    #     for b in range(B):
    #         ids_b = pruned_ids_list[b]
    #         Lb = len(ids_b)
    #         pruned_context_ids[b, :Lb] = torch.tensor(ids_b, device=device, dtype=context_ids.dtype)
    #         ctx_mask[b, :Lb] = 1

    #     return pruned_context_ids, pruned_spans_list, ctx_mask
    # ✅ 修改为：同时返回pruned_chunk_labels
@staticmethod
def _prune_context(
    context_ids: torch.Tensor,
    chunk_spans: List[List[Tuple[int,int]]],
    chunk_scores: List[torch.Tensor],
    topk_chunks: int,
    device: torch.device,
    chunk_labels: Optional[List[torch.Tensor]] = None,  # 新增参数
) -> Tuple[torch.Tensor, List[List[Tuple[int,int]]], torch.Tensor, Optional[List[torch.Tensor]]]:
    """返回元组：(pruned_ids, pruned_spans, mask, pruned_labels)"""
    B, K_full = context_ids.shape
    pruned_ids_list, pruned_spans_list = [], []
    pruned_labels_list = [] if chunk_labels is not None else None  # 新增

    for b in range(B):
        spans_b = chunk_spans[b] if b < len(chunk_spans) else []
        scores_b = chunk_scores[b] if b < len(chunk_scores) else None
        labels_b = chunk_labels[b] if chunk_labels is not None and b < len(chunk_labels) else None

        # fallback逻辑保持不变
        if not spans_b or scores_b is None or scores_b.numel() == 0:
            ids_b = context_ids[b].tolist()
            pruned_ids_list.append(ids_b)
            pruned_spans_list.append([(0, len(ids_b))])
            if chunk_labels is not None:
                pruned_labels_list.append(torch.tensor([0], device=device, dtype=torch.long))
            continue

        J = len(spans_b)
        k_eff = min(topk_chunks, J)
        top_idx = torch.topk(scores_b, k=k_eff).indices.tolist()
        top_idx = sorted(top_idx, key=lambda i: float(scores_b[i]), reverse=True)

        new_ids_b, new_spans_b = [], []
        new_labels_b = [] if chunk_labels is not None else None
        cur = 0
        for j in top_idx:
            s, e = spans_b[j]
            s_clamped = max(0, min(K_full, s))
            e_clamped = max(s_clamped, min(K_full, e))
            if e_clamped <= s_clamped: continue
            span_len = e_clamped - s_clamped
            new_spans_b.append((cur, cur + span_len))
            new_ids_b.extend(context_ids[b, s_clamped:e_clamped].tolist())
            cur += span_len
            
            if chunk_labels is not None and labels_b is not None:
                new_labels_b.append(int(labels_b[j]))  # 记录选中chunk的label

        if not new_ids_b:
            new_ids_b = context_ids[b].tolist()
            new_spans_b = [(0, len(new_ids_b))]
            if chunk_labels is not None:
                new_labels_b = [int(labels_b[0]) if labels_b is not None and len(labels_b) > 0 else 0]

        pruned_ids_list.append(new_ids_b)
        pruned_spans_list.append(new_spans_b)
        if chunk_labels is not None:
            pruned_labels_list.append(torch.tensor(new_labels_b, device=device, dtype=torch.long))

    # padding逻辑保持不变...
    max_len = max(len(ids) for ids in pruned_ids_list) if pruned_ids_list else 0
    pruned_context_ids = context_ids.new_full((B, max_len), fill_value=0)
    ctx_mask = context_ids.new_zeros((B, max_len), dtype=torch.long)

    for b in range(B):
        ids_b = pruned_ids_list[b]
        Lb = len(ids_b)
        pruned_context_ids[b, :Lb] = torch.tensor(ids_b, device=device, dtype=context_ids.dtype)
        ctx_mask[b, :Lb] = 1

    # ✅ 新增返回pruned_labels_list
    return pruned_context_ids, pruned_spans_list, ctx_mask, pruned_labels_list



    def forward(
        self,
        question_ids: torch.Tensor,
        context_ids: torch.Tensor,
        answer_ids: Optional[torch.Tensor] = None,
        chunk_spans: Optional[List[List[Tuple[int,int]]]] = None,
        chunk_labels: Optional[List[torch.Tensor]] = None,
        topk_chunks: int = 2,
    ) -> Dict[str, torch.Tensor]:
        
        device = question_ids.device
        B, Q = question_ids.shape
        if chunk_spans is None: chunk_spans = [[] for _ in range(B)]

        # 1) Embedding
        q_emb_full = self.llama.embed_tokens(question_ids)
        c_emb_full = self.llama.embed_tokens(context_ids)

        # 2) Retrieval
        retrieval_out = self.retrieval(q_emb_full, c_emb_full, chunk_spans)
        final_scores = retrieval_out["final_scores"]

        # 3) Pruning
        # pruned_context_ids, _, ctx_mask = self._prune_context(
        #     context_ids, chunk_spans, final_scores, topk_chunks, device
        # )
        # c_emb_pruned = self.llama.embed_tokens(pruned_context_ids)
        pruned_context_ids, pruned_chunk_spans, ctx_mask, pruned_chunk_labels = self._prune_context(
            context_ids=context_ids,
            chunk_spans=chunk_spans,
            chunk_scores=final_scores,
            topk_chunks=topk_chunks,
            device=device,
            chunk_labels=chunk_labels,  # 传入原始labels
        )
        c_emb_pruned = self.llama.embed_tokens(pruned_context_ids)

        # 4) Fusion -> Prefix
        prefix_latent, attn_info = self.double_stack(q_emb_full, c_emb_pruned)
        cross_weights = attn_info.get("cross_weights")
        # ✅ Top-k soft lexical projection
        with torch.no_grad():
            # 确保 LLaMA embedding 不被更新（你本来就冻结了，但这里更保险）
            embed_w = self.llama.embed_tokens.weight
        prefix_lex, lex_aux = self.lex_proj(prefix_latent, embed_w)
        # if prefix.size(1) > self.prompt_max_len:
        #     prefix = prefix[:, -self.prompt_max_len:, :]
        # 用 lexicalized prefix 替代原 prefix
        prefix = prefix_lex
        B, M, H = prefix.shape

        # === LCA Loss Calculation ===
        lca_loss = torch.tensor(0.0, device=device)
        if self.training and pruned_chunk_labels is not None and self.lca_loss_weight > 0:
            # 从fusion_info获取cross_weights
            cross_weights = attn_info.get("cross_weights")  # [B, heads, M, K]      
            
            if cross_weights is not None:
                # ✅ 关键：detach KV，阻断梯度回传retrieval
                kv_for_lca = c_emb_pruned.detach()
                
                # 使用ZeroForwardLCASLoss
                lca_loss = self.lca_loss_fct(
                    prefix_slots=prefix_latent,  # 使用投影前的latent表示
                    cross_weights=cross_weights,
                    kv_embeddings=kv_for_lca,
                    chunk_spans=pruned_chunk_spans,
                    chunk_labels=pruned_chunk_labels,
                ) * self.lca_loss_weight

        # === Inference Return ===
        if answer_ids is None:
            return {
                "prefix_emb": prefix,
                "pruned_context_ids": pruned_context_ids,
                "pruned_context_emb": c_emb_pruned,
                "question_emb": q_emb_full, # 返回 Query Embedding 供推理拼接
                "chunk_scores": final_scores,
                "lca_loss": lca_loss
            }

        # === Training ===
        ans_in = answer_ids[:, :-1]
        ans_tgt = answer_ids[:, 1:]
        ans_emb = self.llama.embed_tokens(ans_in)

        # 拼接: [Prefix, Pruned_Context, Question, Answer]
        # 注意：这里显式把 q_emb_full 加回去了
        inputs_embeds = torch.cat([prefix, c_emb_pruned, q_emb_full, ans_emb], dim=1)
        total_len = inputs_embeds.size(1)

        # Masks
        prefix_mask = torch.ones(B, M, dtype=torch.long, device=device)
        q_mask = torch.ones(B, Q, dtype=torch.long, device=device)
        ans_mask = (ans_in != self.pad_id).long()
        attention_mask = torch.cat([prefix_mask, ctx_mask, q_mask, ans_mask], dim=1)

        llama_out = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=False
        )
        logits = self.lm_head(llama_out[0])

        # Labels
        labels_prefix = torch.full((B, M), -100, dtype=torch.long, device=device)
        labels_ctx = torch.full_like(ctx_mask, -100)
        labels_q = torch.full((B, Q), -100, dtype=torch.long, device=device)
        labels_ans = torch.where(ans_tgt != self.pad_id, ans_tgt, torch.full_like(ans_tgt, -100))
        
        labels_for_ce = torch.cat([labels_prefix, labels_ctx, labels_q, labels_ans], dim=1)

        # Calc Losses
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_labels = labels_for_ce.reshape(-1)
        
        gen_loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100) if (flat_labels != -100).sum() > 0 else torch.tensor(0.0, device=device)
        # chunk_loss = compute_chunk_loss(final_scores, chunk_labels, device) * self.chunk_loss_weight
        chunk_loss = compute_chunk_loss(self,chunk_scores = final_scores, chunk_labels = chunk_labels,margin=0.1,ce_weight=0) * self.chunk_loss_weight
        total_loss = gen_loss + chunk_loss + (self.lca_loss_weight * lca_loss)

        return {
            "loss": total_loss,
            "total_loss": total_loss,
            "gen_loss": gen_loss,
            "chunk_loss": chunk_loss,
            "lca_loss": lca_loss,
            "chunk_scores": final_scores
        }

