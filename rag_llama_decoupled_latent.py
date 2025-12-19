# rag_llama_decoupled_latent.py
import torch
import torch.nn as nn

from rag_llama_decoupled import RAGLlamaDecoupled
from latent_prefix_fusion import LatentPrefixAttentionStack


class RAGLlamaDecoupledLatent(RAGLlamaDecoupled):
    """
    使用 Latent Prefix Attention 的 RAG 模型
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # === 替换 double_stack ===
        self.double_stack = LatentPrefixAttentionStack(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_layers=self.num_da_layers,
            prefix_len=self.prompt_max_len,     # ✅ 固定长度 prefix
            dropout=self.dropout,
            include_question_in_kv=True,        # 可关以省显存
        )
        

    def _build_prefix(
        self,
        q_emb_full,
        c_emb_pruned,
        q_mask=None,
        c_mask=None,
    ):
        prefix, info = self.double_stack(
            question=q_emb_full,
            context=c_emb_pruned,
            question_mask=q_mask,
            context_mask=c_mask,
        )
        return prefix, info
    # def forward(self, ...):
    #     # 1. 主路径计算
    #     prefix_latent, fusion_info = self.double_stack(q_emb_full, c_emb_pruned)
    #     cross_weights = fusion_info["cross_weights"]  # [B, heads, M, K]
        
    #     # 2. ✅ 立即detach KV，阻断LCA梯度回传retrieval
    #     kv_for_lca = c_emb_pruned.detach()  # 关键：防止LCA干扰retrieval
        
    #     # 3. 词法投影（如果启用）
    #     if hasattr(self, 'lex_proj'):
    #         with torch.no_grad():
    #             embed_w = self.llama.embed_tokens.weight
    #         prefix_lex, _ = self.lex_proj(prefix_latent, embed_w)
    #         # 确保prefix_lex的梯度来自lex_proj，不影响fusion
    #         prefix = prefix_lex
    #     else:
    #         prefix = prefix_latent
        
    #     # 4. ✅ 计算LCA（零额外forward）
    #     lca_loss = self.compute_lca_from_attn(
    #         prefix_slots=prefix_latent,  # 使用投影前表示，保持一致性
    #         cross_weights=cross_weights,
    #         kv_embeddings=kv_for_lca,
    #         chunk_spans=pruned_chunk_spans,
    #         chunk_labels=chunk_labels,
    #     )
    def forward(self, *args, **kwargs):
    # 直接调用父类forward，但替换double_stack
    # 需要重写forward或修改父类逻辑
    # 更简单的方法：在训练脚本中直接使用RAGLlamaDecoupled即可
        return super().forward(*args, **kwargs)