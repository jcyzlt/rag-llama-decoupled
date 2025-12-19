import torch
import torch.nn as nn
import torch.nn.functional as F


# class TopKSoftLexicalProjector(nn.Module):
#     """
#     将 prefix_latent (B,M,H) 投影到词表 embedding 的 top-k 凸组合：
#       logits = W(prefix) -> (B,M,V)
#       idx = topk(logits,k) -> (B,M,k)
#       p = softmax(logits_topk / tau) 
#       prefix_lex = sum_i p_i * E[idx_i]  -> (B,M,H)

#     只训练 W（和可选 tau），E=embed_tokens.weight 冻结且不更新。
#     """
#     def __init__(
#         self,
#         hidden_size: int,
#         vocab_size: int,
#         k: int = 64,
#         tau_init: float = 1.0,
#         learnable_tau: bool = False,
#         dropout: float = 0.0,
#         entropy_reg: float = 0.0,  # 可选：防止过早变尖（建议 0~1e-3）
#     ):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.vocab_size = vocab_size
#         self.k = k
#         self.entropy_reg = entropy_reg
#         self.drop = nn.Dropout(dropout)

#         self.to_vocab = nn.Linear(hidden_size, vocab_size, bias=False)

#         if learnable_tau:
#             self.tau = nn.Parameter(torch.tensor(float(tau_init)))
#         else:
#             self.register_buffer("tau", torch.tensor(float(tau_init)), persistent=False)

#     def forward(self, prefix_latent: torch.Tensor, embed_weight: torch.Tensor):
#         """
#         prefix_latent: (B,M,H)
#         embed_weight: (V,H)  通常传 model.llama.embed_tokens.weight
#         returns:
#           prefix_lex: (B,M,H)
#           aux: dict with entropy_reg_loss (optional), topk_idx (optional for debug)
#         """
#         B, M, H = prefix_latent.shape
#         V, H2 = embed_weight.shape
#         assert H == H2, f"hidden mismatch: prefix {H} vs embed {H2}"
#         assert V == self.vocab_size, f"vocab mismatch: expected {self.vocab_size}, got {V}"

#         x = self.drop(prefix_latent)
#         logits = self.to_vocab(x)  # (B,M,V)

#         k = min(self.k, self.vocab_size)
#         topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)  # (B,M,k), (B,M,k)

#         tau = torch.clamp(self.tau, 0.05, 10.0)
#         p = F.softmax(topk_vals / tau, dim=-1)  # (B,M,k)

#         # 取出 top-k 对应的 embedding: (B,M,k,H)
#         # embed_weight[topk_idx] 会广播索引出 (B,M,k,H)
#         E_topk = embed_weight[topk_idx]  # (B,M,k,H)

#         # 加权求和得到 lexicalized prefix: (B,M,H)
#         prefix_lex = torch.sum(p.unsqueeze(-1) * E_topk, dim=-2)

#         aux = {"topk_idx": topk_idx}

#         # 可选：熵正则（鼓励平滑 or 防止过早尖锐）
#         if self.entropy_reg > 0:
#             ent = -(p * torch.log(p.clamp_min(1e-9))).sum(dim=-1)  # (B,M)
#             # 你可以选择最大化熵（更平滑）=> loss = -ent
#             entropy_loss = -ent.mean()
#             aux["entropy_loss"] = entropy_loss * self.entropy_reg
#         else:
#             aux["entropy_loss"] = None

#         return prefix_lex, aux


class MemoryEfficientTopKProjector(nn.Module):
    def __init__(self, hidden_size, vocab_size, k=64, block_size=1024):
        super().__init__()
        self.to_vocab = nn.Linear(hidden_size, vocab_size, bias=False)
        self.k = k
        self.block_size = block_size  # 每块大小
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))
        
    def forward(self, prefix_latent, embed_weight):
        B, M, H = prefix_latent.shape
        V = embed_weight.shape[0]
        
        # 分块计算topk，避免一次性分配[B,M,V]
        all_vals, all_idx = [], []
        
        for start in range(0, V, self.block_size):
            end = min(V, start + self.block_size)
            
            # 只计算当前块的logits: [B,M,block_size]
            block_logits = self.to_vocab(prefix_latent)[:, :, start:end]
            
            # 块内topk
            block_val, block_idx = torch.topk(
                block_logits, 
                k=min(self.k, block_logits.shape[-1]), 
                dim=-1
            )
            all_vals.append(block_val)
            all_idx.append(block_idx + start)
        
        # 合并所有块: [B,M,blocks*k]
        combined_vals = torch.cat(all_vals, dim=-1)
        combined_idx = torch.cat(all_idx, dim=-1)
        
        # 全局topk
        final_vals, final_pos = torch.topk(combined_vals, k=self.k, dim=-1)
        final_idx = torch.gather(combined_idx, -1, final_pos)
        
        # softmax + 加权
        p = F.softmax(final_vals / self.tau, dim=-1)
        E_topk = embed_weight[final_idx]  # [B,M,k,H]
        prefix_lex = (p.unsqueeze(-1) * E_topk).sum(dim=-2)
        
        return prefix_lex, {"topk_idx": final_idx}