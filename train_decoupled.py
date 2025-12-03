# train_decoupled.py
# 解耦版 RAG 训练：RAGLlamaDecoupled
# - 单次 forward 内部完成：
#   full context → MultiViewChunkRetrieval(评估器) → 剪枝 → pruned context
#   pruned context + question → DoubleAttentionStack(融合器) → prefix
#   prefix + pruned context + answer → LLaMA → gen_loss + chunk_loss

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

from rag_llama_decoupled import RAGLlamaDecoupled  # 你的新模型


# ========== 参数 ==========
def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train RAGLlamaDecoupled (chunk-aware, decoupled retriever/fuser)"
    )
    # 训练超参
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)

    # 路径
    parser.add_argument(
        "--llama_path",
        type=str,
        default="/your/path/to/llama-2-7b",   # 换成你自己的
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./cleaned_train_direction_a.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./trained_decoupled",
    )

    # DoubleAttention 配置
    parser.add_argument("--da_layers", type=int, default=1)
    parser.add_argument("--da_heads", type=int, default=16)
    parser.add_argument("--da_dropout", type=float, default=0.1)
    parser.add_argument("--prompt_max_len", type=int, default=32)

    # 损失权重
    parser.add_argument("--chunk_loss_weight", type=float, default=0.1)

    # 评估器选多少个 chunk
    parser.add_argument("--topk_chunks", type=int, default=4)

    # 随机种子
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ========== 工具 ==========
def set_seed(seed: int):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ========== 数据集 ==========
class RAGDecoupledDataset(Dataset):
    """
    预处理后的数据格式和 Direction A 保持一致：
      [
        {
          "id": str,
          "question": str,
          "answer": str,
          "docs": [str, ...],        # 每个 doc = 一个 chunk
          "doc_labels": [0/1, ...]   # 该 doc 是否属于 supporting_facts
        },
        ...
      ]

    这里：
      - 每个 doc 对应一个 chunk
      - 按 doc 逐个 tokenize、拼成一个 context_ids，并记录每个 doc 的 (s,e) 作为 chunk_spans
      - doc_labels 用作 chunk_labels，用于 chunk 监督
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_question_len: int = 512,
        max_context_len: int = 1024,
        max_answer_len: int = 512,
    ):
        # 兼容 json / jsonl
        self.data = []
        if data_path.endswith(".jsonl"):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.data.append(json.loads(line))
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_question_len = max_question_len
        self.max_context_len = max_context_len
        self.max_answer_len = max_answer_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        q_text = item["question"]
        a_text = item["answer"]
        docs = item["docs"]
        doc_labels = item["doc_labels"]

        # 补 eos
        if self.tokenizer.eos_token and not a_text.endswith(self.tokenizer.eos_token):
            a_text = a_text + self.tokenizer.eos_token

        # ---- 问题 ----
        q_ids = self.tokenizer(
            q_text,
            max_length=self.max_question_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # ---- 上下文：按 doc 逐个 tokenize，再拼接，并记录 span ----
        all_ctx_ids: List[int] = []
        chunk_spans: List[Tuple[int, int]] = []
        kept_labels: List[int] = []

        max_ctx = self.max_context_len
        pad_id = self.tokenizer.pad_token_id

        for doc_text, lab in zip(docs, doc_labels):
            enc = self.tokenizer(
                doc_text,
                max_length=max_ctx,
                truncation=True,
                padding=False,
                add_special_tokens=False,
                return_tensors="pt",
            )
            doc_ids = enc.input_ids.squeeze(0)  # [L_doc]
            L_doc = doc_ids.size(0)
            if L_doc == 0:
                continue

            if len(all_ctx_ids) + L_doc > max_ctx:
                # 放不下了，直接截掉后面的 doc
                break

            s = len(all_ctx_ids)
            e = s + L_doc
            all_ctx_ids.extend(doc_ids.tolist())
            chunk_spans.append((s, e))
            kept_labels.append(int(lab))

        if not all_ctx_ids:
            all_ctx_ids = [pad_id]
            chunk_spans = [(0, 1)]
            kept_labels = [0]

        ctx_ids = torch.full(
            (self.max_context_len,),
            pad_id,
            dtype=torch.long,
        )
        ctx_ids[: len(all_ctx_ids)] = torch.tensor(all_ctx_ids, dtype=torch.long)

        chunk_labels = torch.tensor(kept_labels, dtype=torch.long)  # [J_b]

        # ---- 答案 ----
        a_ids = self.tokenizer(
            a_text,
            max_length=self.max_answer_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).input_ids.squeeze(0)

        return {
            "question_ids": q_ids,
            "context_ids": ctx_ids,
            "answer_ids": a_ids,
            "chunk_spans": chunk_spans,
            "chunk_labels": chunk_labels,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    q_ids = torch.stack([b["question_ids"] for b in batch], dim=0)
    c_ids = torch.stack([b["context_ids"] for b in batch], dim=0)
    a_ids = torch.stack([b["answer_ids"] for b in batch], dim=0)
    chunk_spans = [b["chunk_spans"] for b in batch]
    chunk_labels = [b["chunk_labels"] for b in batch]
    return {
        "question_ids": q_ids,
        "context_ids": c_ids,
        "answer_ids": a_ids,
        "chunk_spans": chunk_spans,
        "chunk_labels": chunk_labels,
    }


# ========== 训练入口 ==========
def main():
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(mixed_precision="fp16", split_batches=False)
    device = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        print(f"[accelerate] mixed_precision = {accelerator.mixed_precision}")
        print(f"[accelerate] device = {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.llama_path)

    # LLaMA 通常没有 pad_token，这里显式加一个
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # ---- dataset & dataloader ----
    dataset = RAGDecoupledDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_question_len=512,
        max_context_len=1024,
        max_answer_len=512,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ---- model ----
    model = RAGLlamaDecoupled(
        llama_model_name_or_path=args.llama_path,
        num_da_layers=args.da_layers,
        num_heads=args.da_heads,
        dropout=args.da_dropout,
        prompt_max_len=args.prompt_max_len,
        chunk_loss_weight=args.chunk_loss_weight,
    )

    # tokenizer 加了 pad_token，需要 resize embedding / lm_head
    model.llama.resize_token_embeddings(len(tokenizer))
    model.resize_output_embeddings(len(tokenizer))

    model.llama.config.pad_token_id = tokenizer.pad_token_id
    model.llama.config.eos_token_id = tokenizer.eos_token_id
    model.pad_id = tokenizer.pad_token_id
    model.eos_id = tokenizer.eos_token_id

    model.llama.config.use_cache = False
    model.llama.gradient_checkpointing_enable()

    if is_main:
        print("model loaded")
        print("  llama dtype:", next(model.llama.parameters()).dtype)
        print("  double_stack dtype:", next(model.double_stack.parameters()).dtype)
        print("  lm_head dtype:", next(model.lm_head.parameters()).dtype)

    # ---- loss & optimizer ----
    # gen_loss 在模型内部已经算好，这里只为了兼容不再额外定义 criterion
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    steps_per_epoch = math.ceil(len(dataset) / (args.batch_size * accelerator.num_processes))
    total_steps = steps_per_epoch * args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    from tqdm import tqdm

    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        sum_gen = 0.0
        sum_chunk = 0.0
        sum_total = 0.0

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            disable=not is_main,
        )

        for step, batch in enumerate(pbar):
            optimizer.zero_grad(set_to_none=True)

            q_ids = batch["question_ids"].to(device)
            c_ids = batch["context_ids"].to(device)
            a_ids = batch["answer_ids"].to(device)
            chunk_spans = batch["chunk_spans"]
            chunk_labels = batch["chunk_labels"]

            # 单次 forward 内部已经完成:
            # - full context 检索 + 剪枝
            # - 融合 + prefix
            # - prefix + pruned ctx + answer → gen_loss & chunk_loss
            with accelerator.autocast():
                out: Dict[str, Any] = model(
                    question_ids=q_ids,
                    context_ids=c_ids,
                    answer_ids=a_ids,
                    chunk_spans=chunk_spans,
                    chunk_labels=chunk_labels,
                    topk_chunks=args.topk_chunks,
                )

                total_loss = out["total_loss"]
                gen_loss = out["gen_loss"]
                chunk_loss = out["chunk_loss"]

            total_loss = total_loss.float()
            accelerator.backward(total_loss)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()

            sum_gen += gen_loss.detach().float().item()
            sum_chunk += chunk_loss.detach().float().item()
            sum_total += total_loss.detach().float().item()
            global_step += 1

            pbar.set_postfix({
                "g_ls": f"{sum_gen / (step + 1):.4f}",
                "c_ls": f"{sum_chunk / (step + 1):.4f}",
                "t_ls": f"{sum_total / (step + 1):.4f}",
            })

        if is_main:
            n = len(dataloader)
            print(f"\nEpoch {epoch + 1} avg loss:")
            print(f"  gen_loss  : {sum_gen / n:.4f}")
            print(f"  chunk_loss: {sum_chunk / n:.4f}")
            print(f"  total     : {sum_total / n:.4f}")

    # ---- save ----
    if is_main:
        unwrapped = accelerator.unwrap_model(model)
        save_dir = os.path.join(args.output_dir, "final_model_decoupled")
        unwrapped.save_pretrained(save_dir, safe_serialization=False)
        tokenizer.save_pretrained(save_dir)
        print(f"model saved to: {save_dir}")


if __name__ == "__main__":
    main()
