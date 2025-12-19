# # train_decoupled.py
# # 解耦版 RAG 训练：RAGLlamaDecoupled
# # - 单次 forward 内部完成：
# #   full context → MultiViewChunkRetrieval(评估器) → 剪枝 → pruned context
# #   pruned context + question → DoubleAttentionStack(融合器) → prefix
# #   prefix + pruned context + answer → LLaMA → gen_loss + chunk_loss

# import os
# import math
# import json
# import random
# from typing import Dict, Any, List, Tuple

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoTokenizer, get_linear_schedule_with_warmup
# from accelerate import Accelerator


# from rag_llama_decoupled import RAGLlamaDecoupled  # 你的新模型


# # ========== 参数 ==========
# def parse_args():
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="Train RAGLlamaDecoupled (chunk-aware, decoupled retriever/fuser)"
#     )
#     # 训练超参
#     parser.add_argument("--batch_size", type=int, default=2)
#     parser.add_argument("--epochs", type=int, default=5)
#     parser.add_argument("--lr", type=float, default=1e-4)

#     # 路径
#     parser.add_argument(
#         "--llama_path",
#         type=str,
#         default="/media/hc-sfxz/4738C1D329F4278F/zlt/version6/TinyLlama-1.1B-Chat-v1.0",   # 换成你自己的
#     )
#     parser.add_argument(
#         "--data_path",
#         type=str,
#         default="/media/hc-sfxz/4738C1D329F4278F/zlt/version6/cleaned_data.jsonl",
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default="./trained_decoupled",
#     )

#     parser.add_argument(
#         "--stage",
#         type=int,
#         default=1,
#         choices=[1, 2],
#         help="1: 只用局部评估器训练(local-only)，2: 用local+global+router继续训练",
#     )
#     parser.add_argument(
#         "--resume_from",
#         type=str,
#         default=None,
#         help="stage=2 时，从 stage1 的输出目录加载模型",
#     )

#     # DoubleAttention 配置
#     parser.add_argument("--da_layers", type=int, default=1)
#     parser.add_argument("--da_heads", type=int, default=16)
#     parser.add_argument("--da_dropout", type=float, default=0.1)
#     parser.add_argument("--prompt_max_len", type=int, default=32)

#     # 损失权重
#     parser.add_argument("--chunk_loss_weight", type=float, default=1.0)

#     # 评估器选多少个 chunk
#     parser.add_argument("--topk_chunks", type=int, default=3)

#     # 随机种子
#     parser.add_argument("--seed", type=int, default=42)

#     return parser.parse_args()


# # ========== 工具 ==========
# def set_seed(seed: int):
#     import numpy as np
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# # ========== 数据集 ==========
# class RAGDecoupledDataset(Dataset):
#     """
#     预处理后的数据格式和 Direction A 保持一致：
#       [
#         {
#           "id": str,
#           "question": str,
#           "answer": str,
#           "docs": [str, ...],        # 每个 doc = 一个 chunk
#           "doc_labels": [0/1, ...]   # 该 doc 是否属于 supporting_facts
#         },
#         ...
#       ]

#     这里：
#       - 每个 doc 对应一个 chunk
#       - 按 doc 逐个 tokenize、拼成一个 context_ids，并记录每个 doc 的 (s,e) 作为 chunk_spans
#       - doc_labels 用作 chunk_labels，用于 chunk 监督
#     """

#     def __init__(
#         self,
#         data_path: str,
#         tokenizer: AutoTokenizer,
#         max_question_len: int = 512,
#         max_context_len: int = 1024,
#         max_answer_len: int = 512,
#     ):
#         # 兼容 json / jsonl
#         self.data = []
#         if data_path.endswith(".jsonl"):
#             with open(data_path, "r", encoding="utf-8") as f:
#                 for line in f:
#                     line = line.strip()
#                     if not line:
#                         continue
#                     self.data.append(json.loads(line))
#         else:
#             with open(data_path, "r", encoding="utf-8") as f:
#                 self.data = json.load(f)

#         self.tokenizer = tokenizer
#         self.max_question_len = max_question_len
#         self.max_context_len = max_context_len
#         self.max_answer_len = max_answer_len

#     def __len__(self) -> int:
#         return len(self.data)

#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         item = self.data[idx]
#         q_text = item["question"]
#         a_text = item["answer"]
#         docs = item["docs"]
#         doc_labels = item["doc_labels"]

#         # === 新增：答案扩充 (Data Augmentation for Short Answers) ===
#         # 简单的 heuristic: 如果答案很短（比如少于 10 个词），就套模板
#         # 或者无脑套模板，让模型学会生成 "The answer is ..."
#         # 你可以根据实际情况调整这个逻辑
#         if len(a_text.split()) < 10: 
#             # 移除可能存在的标点，防止 "Paris." -> "The answer is Paris.."
#             clean_a = a_text.rstrip(".")
#             # 模板化
#             a_text = f"The answer is {clean_a}."


#         # 补 eos
#         if self.tokenizer.eos_token and not a_text.endswith(self.tokenizer.eos_token):
#             a_text = a_text + self.tokenizer.eos_token

#         # ---- 问题 ----
#         q_ids = self.tokenizer(
#             q_text,
#             max_length=self.max_question_len,
#             truncation=True,
#             padding="max_length",
#             return_tensors="pt",
#         ).input_ids.squeeze(0)

#         # ---- 上下文：按 doc 逐个 tokenize，再拼接，并记录 span ----
#         all_ctx_ids: List[int] = []
#         chunk_spans: List[Tuple[int, int]] = []
#         kept_labels: List[int] = []

#         max_ctx = self.max_context_len
#         pad_id = self.tokenizer.pad_token_id

#         for doc_text, lab in zip(docs, doc_labels):
#             enc = self.tokenizer(
#                 doc_text,
#                 max_length=max_ctx,
#                 truncation=True,
#                 padding=False,
#                 add_special_tokens=False,
#                 return_tensors="pt",
#             )
#             doc_ids = enc.input_ids.squeeze(0)  # [L_doc]
#             L_doc = doc_ids.size(0)
#             if L_doc == 0:
#                 continue

#             if len(all_ctx_ids) + L_doc > max_ctx:
#                 # 放不下了，直接截掉后面的 doc
#                 break

#             s = len(all_ctx_ids)
#             e = s + L_doc
#             all_ctx_ids.extend(doc_ids.tolist())
#             chunk_spans.append((s, e))
#             kept_labels.append(int(lab))

#         if not all_ctx_ids:
#             all_ctx_ids = [pad_id]
#             chunk_spans = [(0, 1)]
#             kept_labels = [0]

#         ctx_ids = torch.full(
#             (self.max_context_len,),
#             pad_id,
#             dtype=torch.long,
#         )
#         ctx_ids[: len(all_ctx_ids)] = torch.tensor(all_ctx_ids, dtype=torch.long)

#         chunk_labels = torch.tensor(kept_labels, dtype=torch.long)  # [J_b]

#         # # Debug 模式（先临时这样写）
#         # J = len(chunk_spans)
#         # pseudo_labels = [0] * J
#         # if J > 0:
#         #     pseudo_labels[0] = 1   # 强行让第0个chunk是正例

#         # chunk_labels = torch.tensor(pseudo_labels, dtype=torch.long)
        
        
#         # ---- 答案 ----
#         a_ids = self.tokenizer(
#             a_text,
#             max_length=self.max_answer_len,
#             truncation=True,
#             padding="max_length",
#             return_tensors="pt",
#         ).input_ids.squeeze(0)

#         return {
#             "question_ids": q_ids,
#             "context_ids": ctx_ids,
#             "answer_ids": a_ids,
#             "chunk_spans": chunk_spans,
#             "chunk_labels": chunk_labels,
#         }


# def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
#     q_ids = torch.stack([b["question_ids"] for b in batch], dim=0)
#     c_ids = torch.stack([b["context_ids"] for b in batch], dim=0)
#     a_ids = torch.stack([b["answer_ids"] for b in batch], dim=0)
#     chunk_spans = [b["chunk_spans"] for b in batch]
#     chunk_labels = [b["chunk_labels"] for b in batch]
#     return {
#         "question_ids": q_ids,
#         "context_ids": c_ids,
#         "answer_ids": a_ids,
#         "chunk_spans": chunk_spans,
#         "chunk_labels": chunk_labels,
#     }


# # ========== 训练入口 ==========
# def main():
#     args = parse_args()
#     set_seed(args.seed)

#     accelerator = Accelerator(mixed_precision="fp16", split_batches=False)
#     device = accelerator.device
#     is_main = accelerator.is_main_process

#     if is_main:
#         print(f"[accelerate] mixed_precision = {accelerator.mixed_precision}")
#         print(f"[accelerate] device = {device}")

#     os.makedirs(args.output_dir, exist_ok=True)

#     # ---- tokenizer ----
#     tokenizer = AutoTokenizer.from_pretrained(args.llama_path)

#     # LLaMA 通常没有 pad_token，这里显式加一个
#     if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
#         tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

#     # ---- dataset & dataloader ----
#     dataset = RAGDecoupledDataset(
#         data_path=args.data_path,
#         tokenizer=tokenizer,
#         max_question_len=512,
#         max_context_len=1024,
#         max_answer_len=512,
#     )

#     dataloader = DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         drop_last=False,
#         num_workers=4,
#         pin_memory=True,
#         collate_fn=collate_fn,
#     )

#     # ---- model ----
#     # model = RAGLlamaDecoupled(
#     #     llama_model_name_or_path=args.llama_path,
#     #     num_da_layers=args.da_layers,
#     #     num_heads=args.da_heads,
#     #     dropout=args.da_dropout,
#     #     prompt_max_len=args.prompt_max_len,
#     #     chunk_loss_weight=args.chunk_loss_weight,
#     # )
#     # # tokenizer 加了 pad_token，需要 resize embedding / lm_head
#     # model.llama.resize_token_embeddings(len(tokenizer))
#     # model.resize_output_embeddings(len(tokenizer))

#      # ---- model ----
#     if args.stage == 1:
#         # Stage1：local-only 模式，从 LLaMA 权重初始化
#         model = RAGLlamaDecoupled(
#             config=args.llama_path,
#             num_da_layers=args.da_layers,
#             num_heads=args.da_heads,
#             dropout=args.da_dropout,
#             prompt_max_len=args.prompt_max_len,
#             chunk_loss_weight=args.chunk_loss_weight,
#             enable_global=False,   # 核心：第一阶段禁用 global
#         )

#         model.llama.resize_token_embeddings(len(tokenizer))
#         model.resize_output_embeddings(len(tokenizer))

#     else:  # stage == 2
#         assert args.resume_from is not None, "stage=2 需要 --resume_from=stage1_model_dir"

#         # 从 stage1 的 checkpoint 加载（里面 local 已经训好，retrieval.enable_global=False）
#         model = RAGLlamaDecoupled.from_pretrained(
#             args.resume_from,
#             llama_model_name_or_path=args.llama_path,
#             num_da_layers=args.da_layers,
#             num_heads=args.da_heads,
#             dropout=args.da_dropout,
#             prompt_max_len=args.prompt_max_len,
#             chunk_loss_weight=args.chunk_loss_weight,
#             enable_global=True,   # 核心：第二阶段启用 global+router
#         )

#         # embedding resize 保持一致
#         model.llama.resize_token_embeddings(len(tokenizer))
#         model.resize_output_embeddings(len(tokenizer))
        
#         for n, p in model.named_parameters():
#             if "retrieval.local_scorer" in n:
#                 p.requires_grad = False       # 固定 local
#             elif "retrieval.global_scorer" in n:
#                 p.requires_grad = True        # 训练 global
#             elif "retrieval.router" in n:
#                 p.requires_grad = True        # 训练 router

    

#     model.llama.config.pad_token_id = tokenizer.pad_token_id
#     model.llama.config.eos_token_id = tokenizer.eos_token_id
#     model.pad_id = tokenizer.pad_token_id
#     model.eos_id = tokenizer.eos_token_id

#     model.llama.config.use_cache = False
#     model.llama.gradient_checkpointing_enable()

#     if is_main:
#         print("model loaded")
#         print("  llama dtype:", next(model.llama.parameters()).dtype)
#         print("  double_stack dtype:", next(model.double_stack.parameters()).dtype)
#         print("  lm_head dtype:", next(model.lm_head.parameters()).dtype)

#     # ---- loss & optimizer ----
#     # gen_loss 在模型内部已经算好，这里只为了兼容不再额外定义 criterion
#     trainable_params = [p for p in model.parameters() if p.requires_grad]
#     optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

#     steps_per_epoch = math.ceil(len(dataset) / (args.batch_size * accelerator.num_processes))
#     total_steps = steps_per_epoch * args.epochs

#     scheduler = get_linear_schedule_with_warmup(
#         optimizer=optimizer,
#         num_warmup_steps=int(total_steps * 0.1),
#         num_training_steps=total_steps,
#     )

#     model, optimizer, dataloader, scheduler = accelerator.prepare(
#         model, optimizer, dataloader, scheduler
#     )

#     from tqdm import tqdm

#     global_step = 0

#     for epoch in range(args.epochs):
#         model.train()
#         sum_gen = 0.0
#         sum_chunk = 0.0
#         sum_total = 0.0

        

#         pbar = tqdm(
#             dataloader,
#             desc=f"Epoch {epoch + 1}/{args.epochs}",
#             disable=not is_main,
#         )
#         # pbar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process)

#         for step, batch in enumerate(pbar):
#             optimizer.zero_grad(set_to_none=True)

#             q_ids = batch["question_ids"].to(device)
#             c_ids = batch["context_ids"].to(device)
#             a_ids = batch["answer_ids"].to(device)
#             chunk_spans = batch["chunk_spans"]
#             chunk_labels = batch["chunk_labels"]

#             # 单次 forward 内部已经完成:
#             # - full context 检索 + 剪枝
#             # - 融合 + prefix
#             # - prefix + pruned ctx + answer → gen_loss & chunk_loss
#             with accelerator.autocast():
#                 out: Dict[str, Any] = model(
#                     question_ids=q_ids,
#                     context_ids=c_ids,
#                     answer_ids=a_ids,
#                     chunk_spans=chunk_spans,
#                     chunk_labels=chunk_labels,
#                     topk_chunks=args.topk_chunks,
#                 )

#                 total_loss = out["total_loss"]
#                 gen_loss = out["gen_loss"]
#                 chunk_loss = out["chunk_loss"]

#             #     final_scores = out["chunk_scores"]  # List[Tensor[J_b]]，按你的命名
#             #     fs0 = final_scores[0]

#             # print("fs0.requires_grad:", fs0.requires_grad)
#             # print("fs0.grad_fn:", fs0.grad_fn)


#             total_loss = total_loss.float()
#             accelerator.backward(total_loss)


            
#             # # 看看 local_scorer 里权重的梯度是不是零
#             # gn_q = model.retrieval.local_scorer.self_attn.q_proj.weight.grad
#             # gn_k = model.retrieval.local_scorer.self_attn.k_proj.weight.grad
#             # gn_v = model.retrieval.local_scorer.self_attn.v_proj.weight.grad

#             # print("grad q_proj:", None if gn_q is None else gn_q.abs().mean().item())
#             # print("grad k_proj:", None if gn_k is None else gn_k.abs().mean().item())
#             # print("grad v_proj:", None if gn_v is None else gn_v.abs().mean().item())



#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
#             optimizer.step()
#             scheduler.step()

#             sum_gen += gen_loss.detach().float().item()
#             sum_chunk += chunk_loss.detach().float().item()
#             sum_total += total_loss.detach().float().item()
#             global_step += 1

#             pbar.set_postfix({
#                 "g_ls": f"{sum_gen / (step + 1):.4f}",
#                 "c_ls": f"{sum_chunk / (step + 1):.4f}",
#                 "t_ls": f"{sum_total / (step + 1):.4f}",
#             })

#         if is_main:
#             n = len(dataloader)
#             print(f"\nEpoch {epoch + 1} avg loss:")
#             print(f"  gen_loss  : {sum_gen / n:.4f}")
#             print(f"  chunk_loss: {sum_chunk / n:.4f}")
#             print(f"  total     : {sum_total / n:.4f}")

#     # ---- save ----
#     if is_main:
#         unwrapped = accelerator.unwrap_model(model)
#         save_dir = os.path.join(args.output_dir, "final_model_decoupled")
#         unwrapped.save_pretrained(save_dir, safe_serialization=False)
#         tokenizer.save_pretrained(save_dir)
#         print(f"model saved to: {save_dir}")


# if __name__ == "__main__":
#     main()





# train_decoupled.py

import os
import math
import json
import random
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm
import matplotlib.pyplot as plt

# from rag_llama_decoupled import RAGLlamaDecoupled
from rag_llama_decoupled_latent import RAGLlamaDecoupledLatent

# 必须设置
plt.switch_backend('Agg')

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    # 基础
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--seed", type=int, default=42)
    # 路径
    parser.add_argument("--llama_path", type=str, default="/media/hc-sfxz/4738C1D329F4278F/zlt/version6/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--data_path", type=str, default="/media/hc-sfxz/4738C1D329F4278F/zlt/version6/cleaned_data1.jsonl")
    parser.add_argument("--output_dir", type=str, default="./trained_decoupled")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    # 模型配置
    parser.add_argument("--da_layers", type=int, default=2)
    parser.add_argument("--da_heads", type=int, default=8)
    parser.add_argument("--da_dropout", type=float, default=0.1)
    parser.add_argument("--prompt_max_len", type=int, default=32)
    parser.add_argument("--topk_chunks", type=int, default=4)
    # Loss 权重
    parser.add_argument("--chunk_loss_weight", type=float, default=3.0)
    parser.add_argument("--lca_loss_weight", type=float, default=0)

    # 累计多少batch更新
    parser.add_argument("--grad_accum_steps", type=int, default=4)

    
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 检索的指标
def compute_topk_f_metrics(chunk_scores_list, chunk_labels_list, k: int):
    """
    chunk_scores_list: List[Tensor[J_b]]  来自 model 输出
    chunk_labels_list: List[Tensor[J_b]]  batch["chunk_labels"]
    返回: (precision@k, recall@k, F1@k, F2@k) 的 batch 平均值
    """
    precs, recs, f1s, f2s = [], [], [], []

    for scores_b, labels_b in zip(chunk_scores_list, chunk_labels_list):
        if scores_b is None or scores_b.numel() == 0:
            continue
        labels_b = labels_b.to(scores_b.device)
        # 正例总数
        P = labels_b.sum().item()
        if P == 0:
            # 没有正例的样本对检索评价没意义，跳过
            continue

        J = scores_b.size(0)
        k_eff = min(k, J)

        # 取 top-k indices
        topk_idx = torch.topk(scores_b, k=k_eff, dim=0).indices
        topk_labels = labels_b[topk_idx]

        TP = topk_labels.sum().item()

        prec = TP / float(k_eff)
        rec = TP / float(P)

        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0

        beta2 = 4.0  # beta = 2
        if beta2 * prec + rec > 0:
            f2 = (1 + beta2) * prec * rec / (beta2 * prec + rec)
        else:
            f2 = 0.0

        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        f2s.append(f2)

    if not precs:
        return 0.0, 0.0, 0.0, 0.0

    def _mean(x): return float(sum(x) / len(x))

    return _mean(precs), _mean(recs), _mean(f1s), _mean(f2s)


class RAGDecoupledDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=512):
        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): 
                    try: self.data.append(json.loads(line))
                    except: pass
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        q_text = item["question"]
        a_text = item["answer"]
        
                
        if self.tokenizer.eos_token and not a_text.endswith(self.tokenizer.eos_token):
            a_text += self.tokenizer.eos_token

        # Tokenize
        q_ids = self.tokenizer(q_text, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt").input_ids.squeeze(0)
        a_ids = self.tokenizer(a_text, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt").input_ids.squeeze(0)
        
        # Context
        all_ctx_ids, chunk_spans, kept_labels = [], [], []
        max_ctx = 1024
        
        for doc, lab in zip(item["docs"], item["doc_labels"]):
            d_ids = self.tokenizer(doc, truncation=True, max_length=512, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)
            if d_ids.size(0) == 0: continue
            if len(all_ctx_ids) + d_ids.size(0) > max_ctx: break
            
            s = len(all_ctx_ids)
            all_ctx_ids.extend(d_ids.tolist())
            chunk_spans.append((s, s + d_ids.size(0)))
            kept_labels.append(lab)
            
        if not all_ctx_ids:
            all_ctx_ids = [self.tokenizer.pad_token_id]
            chunk_spans = [(0, 1)]
            kept_labels = [0]
            
        ctx_ids = torch.full((max_ctx,), self.tokenizer.pad_token_id, dtype=torch.long)
        ctx_ids[:len(all_ctx_ids)] = torch.tensor(all_ctx_ids, dtype=torch.long)
        
        return {
            "question_ids": q_ids, "context_ids": ctx_ids, "answer_ids": a_ids,
            "chunk_spans": chunk_spans, "chunk_labels": torch.tensor(kept_labels, dtype=torch.long)
        }

def collate_fn(batch):
    return {
        "question_ids": torch.stack([b["question_ids"] for b in batch]),
        "context_ids": torch.stack([b["context_ids"] for b in batch]),
        "answer_ids": torch.stack([b["answer_ids"] for b in batch]),
        "chunk_spans": [b["chunk_spans"] for b in batch],
        "chunk_labels": [b["chunk_labels"] for b in batch]
    }

def main():
    args = parse_args()
    set_seed(args.seed)
    accelerator = Accelerator(mixed_precision="fp16",gradient_accumulation_steps=args.grad_accum_steps)
    os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.llama_path)
    if tokenizer.pad_token_id is None: tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    
    dataset = RAGDecoupledDataset(args.data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    if args.stage == 1:
        model = RAGLlamaDecoupledLatent(
            config=args.llama_path, 
            num_da_layers=args.da_layers, num_heads=args.da_heads, dropout=args.da_dropout,
            prompt_max_len=args.prompt_max_len, chunk_loss_weight=args.chunk_loss_weight,
            lca_loss_weight=args.lca_loss_weight, enable_global=False
        )
        model.llama.resize_token_embeddings(len(tokenizer))
        model.resize_output_embeddings(len(tokenizer))
    else:
        # Stage 2 逻辑省略，按需添加
        pass

    # Sync special tokens
    model.pad_id = tokenizer.pad_token_id
    model.eos_id = tokenizer.eos_token_id
    model.llama.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    loss_history = {
        "step": [],
        "gen": [],
        "chunk": [],
        "lca": [],
        "total": [],
        
    }
    retrieval_history ={
        "step":[],
        "prec": [],     # 新增
        "rec": [],      # 新增
        "f1": [],       # 新增
        "f2": [],       # 新增
    }

    # 检查训练的参数
    trainable = [(n, p.numel()) for n,p in model.named_parameters() if p.requires_grad]
    print("Trainable params:", sum(x[1] for x in trainable))
    print([n for n,_ in trainable[:50]])


    global_step = 0
    
    min_k = 1
    max_k = args.topk_chunks
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=not accelerator.is_main_process)
        
        # 当前 epoch 使用的 top-k（线性增长）
        if args.epochs > 1:
            curr_k = min_k + (max_k - min_k) * epoch // (args.epochs - 1)
        else:
            curr_k = max_k
        curr_k = int(curr_k)
        curr_k = max(1, min(curr_k, max_k))
        
                
        for step, batch in enumerate(pbar):
            # 关键：让 accelerate 知道要累积梯度
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    out = model(
                        question_ids=batch["question_ids"],
                        context_ids=batch["context_ids"],
                        answer_ids=batch["answer_ids"],
                        chunk_spans=batch["chunk_spans"],
                        chunk_labels=batch["chunk_labels"],
                        topk_chunks=curr_k,
                    )
                    loss = out["total_loss"]

            accelerator.backward(loss)

            # 只在累积满一次“有效 batch”时才会真的 step
            if accelerator.sync_gradients:
                optimizer.step()
                optimizer.zero_grad()

            # # ===== 计算 top-k 指标 =====
            # chunk_scores = out["chunk_scores"]          # List[Tensor[J_b]]
            # chunk_labels = batch["chunk_labels"]        # List[Tensor[J_b]]
            # k_eval = args.topk_chunks                   # 你现在就是 4

            # prec_k, rec_k, f1_k, f2_k = compute_topk_f_metrics(chunk_scores, chunk_labels, k_eval)
            

            # ====== logging 一点小改动 ======
            g_loss = out["gen_loss"].item()
            c_loss = out["chunk_loss"].item()
            l_loss = out["lca_loss"].item()
            global_step += 1

            # 这里你可以选择：
            # 1) 每个 micro step 都记一次 loss（现在这样）；
            # 2) 只在 sync_gradients=True 时记（等效大 batch 级别的 loss）
            # pbar.set_postfix({"g": f"{g_loss:.3f}", "c": f"{c_loss:.3f}","P@{k_eval}": f"{prec_k:.2f}","R@{k_eval}": f"{rec_k:.2f}","F1@{k_eval}": f"{f1_k:.2f}","k": curr_k,})
            pbar.set_postfix({"g": f"{g_loss:.3f}", "c": f"{c_loss:.3f}","k": curr_k,})

            if accelerator.is_main_process and global_step % 50 == 0:
                loss_history["step"].append(global_step)
                loss_history["gen"].append(g_loss)
                loss_history["chunk"].append(c_loss)
                # loss_history["lca"].append(l_loss)
                loss_history["total"].append(loss.item())


                # retrieval_history["prec"].append(prec_k)
                # retrieval_history["rec"].append(rec_k)
                # retrieval_history["f1"].append(f1_k)
                # retrieval_history["f2"].append(f2_k)


                plt.figure(figsize=(10,6))
                plt.plot(loss_history["step"], loss_history["gen"], label="Gen")
                plt.plot(loss_history["step"], loss_history["chunk"], label="Chunk")
                plt.plot(loss_history["step"], loss_history["total"], label="Total")
                plt.legend()
                plt.savefig(os.path.join(args.output_dir, "loss_1.png"))
                plt.close()
                

                # plt.figure(figsize=(10,6))
                # plt.plot(loss_history["step"], retrieval_history["prec"], label="Prec")
                # plt.plot(loss_history["step"], retrieval_history["rec"], label="Rec")
                # plt.plot(loss_history["step"], retrieval_history["f1"], label="F1")
                # plt.plot(loss_history["step"], retrieval_history["f2"], label="F2")
                # # plt.plot(loss_history["step"], loss_history["lca"], label="LCA")
                # plt.legend()
                # plt.savefig(os.path.join(args.output_dir, "f1.png"))
                # plt.close()

    if accelerator.is_main_process:
        save_dir = os.path.join(args.output_dir, "final_model_decoupled_1")
        accelerator.unwrap_model(model).save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Saved to {save_dir}")

if __name__ == "__main__":
    main()
