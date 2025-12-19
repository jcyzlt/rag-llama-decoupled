# import torch
# import argparse
# import os
# from typing import List, Tuple
# from transformers import AutoTokenizer, LlamaModel, AutoConfig

# # 引入你的模型定义
# from rag_llama_decoupled import RAGLlamaDecoupled

# def parse_args():
#     parser = argparse.ArgumentParser(description="Inference for Decoupled RAG")
#     parser.add_argument("--model_path", type=str, required=True, help="训练好的模型checkpoint目录 (例如 ./outputs_stage1/final_model_decoupled)")
#     parser.add_argument("--base_model_name", type=str, default="./TinyLlama-1.1B-Chat-v1.0", help="基础LLaMA模型路径(用于加载Tokenizer)")
#     parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
#     parser.add_argument("--max_new_tokens", type=int, default=100)
#     parser.add_argument("--topk_chunks", type=int, default=3, help="推理时保留多少个chunk")
#     return parser.parse_args()

# def run_inference(model, tokenizer, question: str, chunks: List[str], device, topk=2):
#     """
#     运行单次推理流程：
#     1. 编码 Query 和所有 Chunks
#     2. RAG模型内部进行 Retrieval 打分 -> 剪枝 -> Fusion 生成 Prefix
#     3. 拼接 [Prefix, Pruned_Context] (可能还有 Query) -> LLaMA Generate
#     """
#     model.eval()
    
#     # --- 1. 数据准备 ---
#     # Tokenize Question
#     # add_special_tokens=False 因为我们会手动控制拼接
#     q_inputs = tokenizer(question, return_tensors="pt", add_special_tokens=False).to(device)
#     q_ids = q_inputs.input_ids # [1, Q]
    
#     # Tokenize Contexts (Batch Processing)
#     # 简单的处理：直接编码整个 chunk 列表，padding 到最长
#     c_inputs = tokenizer(chunks, return_tensors="pt", padding=True, truncation=True, max_length=256, add_special_tokens=False).to(device)
#     c_ids_batch = c_inputs.input_ids # [Num_Chunks, Seq_Len]
    
#     # RAG模型期望的 context_ids 是 [B, K_full] (即把所有 token 拼成一条长序列)
#     # 同时需要 chunk_spans 来告诉模型哪里是哪个 chunk
#     # 这里我们把所有 chunk 拼成一个由 pad 分隔或者直接拼接的长序列
    
#     # 为了简化，我们把所有 chunk 拼接到一起，并在 chunk_spans 里记录位置
#     flat_context_ids = []
#     chunk_spans = [] # List[Tuple[start, end]]
#     current_idx = 0
    
#     for i in range(len(chunks)):
#         # 获取当前 chunk 的 token ids (去掉 padding)
#         # 注意: c_inputs.attention_mask[i] 可以告诉我们也真实的长度
#         real_len = c_inputs.attention_mask[i].sum().item()
#         token_ids = c_ids_batch[i, :real_len].tolist()
        
#         flat_context_ids.extend(token_ids)
#         chunk_spans.append((current_idx, current_idx + real_len))
#         current_idx += real_len
        
#     # 转为 Tensor [1, Total_Len]
#     context_ids = torch.tensor([flat_context_ids], dtype=torch.long, device=device)
#     # 包装 spans: List[List[Tuple]] -> Batch size = 1
#     batch_chunk_spans = [chunk_spans]

#     print(f"\n{'='*20} Processing {'='*20}")
#     print(f"Question: {question}")
#     print(f"Num Candidates: {len(chunks)}")

#     with torch.no_grad():
#         # --- 2. 调用 RAG 模型的 forward (推理模式) ---
#         # 传入 answer_ids=None，模型会返回 prefix_emb 和 pruned_context_emb
#         outputs = model(
#             question_ids=q_ids,
#             context_ids=context_ids,
#             answer_ids=None,
#             chunk_spans=batch_chunk_spans,
#             topk_chunks=topk
#         )
        
#         # 获取输出
#         prefix_emb = outputs["prefix_emb"]               # [1, M, H]
#         pruned_context_emb = outputs["pruned_context_emb"] # [1, K_pruned, H]
#         final_scores = outputs["chunk_scores"][0]        # [Num_Chunks]
        
#         # --- 3. 打印检索结果 ---
#         print("\n--- Retrieval Scores ---")
#         probs = torch.sigmoid(final_scores) # 转为概率方便看
#         # 获取 topk 索引
#         topk_vals, topk_indices = torch.topk(final_scores, k=min(topk, len(chunks)))
#         topk_set = set(topk_indices.tolist())
        
#         for i, score in enumerate(probs):
#             prefix_str = ">>> SELECTED" if i in topk_set else "    Ignored "
#             # 截取一点内容展示
#             content_preview = chunks[i][:50].replace('\n', ' ')
#             print(f"{prefix_str} Chunk {i}: {score:.4f} | Content: {content_preview}...")

#         # --- 4. 生成 (Generation) ---
#         print("\n--- Generating Answer ---")
        
#         # 构造生成的输入 Embeddings
#         # 你的训练逻辑似乎是: [Prefix, Pruned_Context, Answer]
#         # 推理时，我们应该给 LLaMA: [Prefix, Pruned_Context, Query] (或者根据你的 Prompt 模板)
#         # 通常 RAG 是 Context -> Question -> Answer
        
#         # 获取 Query Embedding (LLaMA 冻结的 embedding)
#         q_emb = model.llama.embed_tokens(q_ids) # [1, Q, H]
        
#         # 拼接: [Prefix, Pruned_Context, Query]
#         # 注意：这里假设你的模型训练时也是这种顺序。如果训练时没有 Query (只靠 Prefix)，则去掉 q_emb
#         # 根据你的 forward 代码: inputs_embeds = torch.cat([prefix, c_emb_pruned, ans_emb], dim=1)
#         # 这意味着训练时 Query 信息已经被 Fusion Module 压缩进 Prefix 了，或者 Query 并没有直接给 Decoder？
#         # 看你的 forward 代码，LLaMA 的输入确实没有 question_ids 对应的 embedding。
#         # 但是！Fusion Module 只是看了 Query。如果 Decoder 看不到 Query 的原文，可能会导致幻觉。
#         # **关键决策**: 通常 RAG 需要把 Query 再拼回去，除非 Prefix 已经极其强大。
#         # 我们这里尝试拼接: [Prefix, Pruned_Context, Query]
        
#         input_embeds = torch.cat([prefix_emb, pruned_context_emb, q_emb], dim=1)
        
#         # 为了让 generate 跑起来，我们需要一个 input_ids 来启动 (通常是 BOS)
#         # 或者 HuggingFace generate 允许只传 inputs_embeds (视版本而定)
#         # 为了稳妥，我们构造一个 dummy input_ids，长度与 embeds 匹配，但实际上 generate 内部主要用 embeds
#         # 更好的方法: 给一个 start_token，然后用 inputs_embeds
        
#         # LLaMA Generate
#         # 注意: LLaMA 的 generate 不直接支持 inputs_embeds 进行自回归生成(这是 HF 的限制)
#         # 但我们可以用 model.llama.generate (如果它是 LlamaForCausalLM)
#         # 你的 model.llama 是 LlamaModel (没有 LM Head)，所以不能直接 generate。
#         # 你在 RAGLlamaDecoupled 里定义了 self.lm_head。
        
#         # === 难点解决 ===
#         # 由于 self.llama 是 LlamaModel (Base Model)，它没有 .generate() 方法。
#         # .generate() 是 LlamaForCausalLM 的方法。
#         # 我们需要手动写一个简单的贪婪搜索 loop，或者临时包装一下。
        
#         # 这里写一个简单的贪婪生成循环 (Greedy Decoding)
#         curr_embeds = input_embeds
#         generated_ids = []
        
#         for _ in range(50): # max_new_tokens
#             # Forward pass
#             # 我们需要构造 attention_mask
#             B, L, H = curr_embeds.shape
#             att_mask = torch.ones(B, L, device=device)
            
#             # 使用你的模型内部组件进行一次 forward
#             # model.llama 输出 hidden_states
#             out = model.llama(inputs_embeds=curr_embeds, attention_mask=att_mask)
#             hidden = out.last_hidden_state # [1, L, H]
            
#             # 取最后一个 token 的 hidden state 预测下一个词
#             last_hidden = hidden[:, -1, :] # [1, H]
#             logits = model.lm_head(last_hidden) # [1, V]
            
#             # Greedy decode
#             next_token_id = torch.argmax(logits, dim=-1) # [1]
            
#             # 停止条件
#             if next_token_id.item() == tokenizer.eos_token_id:
#                 break
            
#             generated_ids.append(next_token_id.item())
            
#             # 准备下一步的 input
#             next_token_emb = model.llama.embed_tokens(next_token_id.unsqueeze(0)) # [1, 1, H]
#             curr_embeds = torch.cat([curr_embeds, next_token_emb], dim=1)
            
#         # 解码
#         text = tokenizer.decode(generated_ids, skip_special_tokens=True)
#         print(f"Generated: {text}")
#         print("="*40)

# def main():
#     args = parse_args()
    
#     print(f"Loading tokenizer: {args.base_model_name}")
#     tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    
#     # 如果 tokenizer 没有 pad_token，通常训练时会添加一个，导致长度+1
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         # 注意：这里我们不需要手动 resize tokenizer，
#         # 因为我们主要关心的是让 Model 的 config 匹配 Checkpoint 的形状

#     print(f"Loading model from: {args.model_path}")
    
#     # --- 关键修改开始 ---
#     # 1. 先加载配置
#     config = AutoConfig.from_pretrained(args.model_path)
    
#     # 2. 【核心修复】检测并修正 vocab_size
#     # 报错信息明确说 checkpoint 是 32001，而 config 可能是 32000
#     # 我们强制把 config 改为 32001，这样模型初始化时就会创建正确大小的矩阵
#     if config.vocab_size == 32000:
#         print("Warning: Config claims vocab_size=32000, but checkpoint requires 32001.")
#         print("Manually adjusting config.vocab_size to 32001 to match checkpoint.")
#         config.vocab_size = 32001

#     # 3. 使用修改后的 config 加载模型
#     try:
#         model = RAGLlamaDecoupled.from_pretrained(args.model_path, config=config)
#     except Exception as e:
#         print(f"Loading failed: {e}")
#         return
#     # --- 关键修改结束 ---

#     model.to(args.device)
#     model.eval()

#     # 预设一个测试案例
#     print("\nRunning Test Case...")
#     test_q = "What is the capital of France?"
#     test_chunks = [
#         "Berlin is the capital of Germany.", 
#         "Paris is the capital and most populous city of France.",
#         "The Eiffel Tower is located in Paris.",
#         "Tokyo is the capital of Japan."
#     ]
    
#     run_inference(model, tokenizer, test_q, test_chunks, args.device, topk=args.topk_chunks)

#     # 交互模式
#     while True:
#         try:
#             print("\n\n--- Interactive Mode (Ctrl+C to exit) ---")
#             q = input("Input Question: ")
#             print("Input Context Chunks (type 'END' on a new line to finish):")
#             chunks = []
#             while True:
#                 line = input("> ")
#                 if line.strip() == "END":
#                     break
#                 if line.strip():
#                     chunks.append(line)
            
#             if not chunks:
#                 print("No chunks provided, skipping.")
#                 continue
                
#             run_inference(model, tokenizer, q, chunks, args.device, topk=args.topk_chunks)
            
#         except KeyboardInterrupt:
#             print("\nExiting...")
#             break
#         except Exception as e:
#             print(f"Error: {e}")
#             import traceback
#             traceback.print_exc()

# if __name__ == "__main__":
#     main()


# import torch
# import argparse
# import os
# from typing import List, Tuple
# from transformers import AutoTokenizer, LlamaModel, AutoConfig

# # 引入你的模型定义
# from rag_llama_decoupled import RAGLlamaDecoupled

# def parse_args():
#     parser = argparse.ArgumentParser(description="Inference for Decoupled RAG (Debug Mode)")
#     parser.add_argument("--model_path", type=str, required=True, help="训练好的模型checkpoint目录")
#     parser.add_argument("--base_model_name", type=str, default="./TinyLlama-1.1B-Chat-v1.0", help="基础LLaMA模型路径")
#     parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
#     parser.add_argument("--max_new_tokens", type=int, default=50)
#     parser.add_argument("--topk_chunks", type=int, default=3)
#     return parser.parse_args()

# def run_inference(model, tokenizer, question: str, chunks: List[str], device, topk=2):
#     model.eval()
    
#     # 1. 数据准备
#     q_inputs = tokenizer(question, return_tensors="pt", add_special_tokens=False).to(device)
#     q_ids = q_inputs.input_ids
    
#     c_inputs = tokenizer(chunks, return_tensors="pt", padding=True, truncation=True, max_length=256, add_special_tokens=False).to(device)
#     c_ids_batch = c_inputs.input_ids 
    
#     # 构造 context_ids
#     flat_context_ids = []
#     chunk_spans = [] 
#     current_idx = 0
#     for i in range(len(chunks)):
#         real_len = c_inputs.attention_mask[i].sum().item()
#         token_ids = c_ids_batch[i, :real_len].tolist()
#         flat_context_ids.extend(token_ids)
#         chunk_spans.append((current_idx, current_idx + real_len))
#         current_idx += real_len
        
#     context_ids = torch.tensor([flat_context_ids], dtype=torch.long, device=device)
#     batch_chunk_spans = [chunk_spans]

#     print(f"\n{'='*20} Processing {'='*20}")
#     print(f"Question: {question}")

#     with torch.no_grad():
#         # 2. RAG Forward (获取特征)
#         outputs = model(
#             question_ids=q_ids,
#             context_ids=context_ids,
#             answer_ids=None,
#             chunk_spans=batch_chunk_spans,
#             topk_chunks=topk
#         )
        
#         prefix_emb = outputs["prefix_emb"]               
#         pruned_context_emb = outputs["pruned_context_emb"] 
#         q_emb = outputs["question_emb"]
#         final_scores = outputs["chunk_scores"][0]        
        
#         # 3. 打印检索情况
#         print("\n--- Retrieval Scores ---")
#         probs = torch.sigmoid(final_scores)
#         topk_vals, topk_indices = torch.topk(final_scores, k=min(topk, len(chunks)))
#         topk_set = set(topk_indices.tolist())
#         for i, score in enumerate(probs):
#             prefix_str = ">>> SELECTED" if i in topk_set else "    Ignored "
#             print(f"{prefix_str} Chunk {i}: {score:.4f} | Content: {chunks[i][:30]}...")

#         # 4. 生成 (Debug Generation)
#         print("\n--- Generating Answer (Debug Mode) ---")
        
        
        
#         # 拼接输入: [Prefix, Pruned_Context, Query]
#         # input_embeds = torch.cat([prefix_emb, pruned_context_emb, q_emb], dim=1)
#         # 选项 B (推荐): 如果训练加了 "The answer is"，推理最好也给个开头
#         prompt_ids = tokenizer("The answer is", return_tensors="pt", add_special_tokens=False).input_ids.to(device)
#         prompt_emb = model.llama.embed_tokens(prompt_ids)
#         input_embeds = torch.cat([prefix_emb, pruned_context_emb, q_emb, prompt_emb], dim=1)
        
        
#         # 检查 NaN
#         if torch.isnan(input_embeds).any():
#             print("!!! WARNING: Input embeddings contain NaN! This will cause garbage output.")
        
#         curr_embeds = input_embeds
#         generated_ids = []
        
#         print(f"Start generating (Max {50} tokens)...")
#         for step in range(50): 
#             B, L, H = curr_embeds.shape
#             att_mask = torch.ones(B, L, device=device)
            
#             # Forward
#             out = model.llama(inputs_embeds=curr_embeds, attention_mask=att_mask)
#             last_hidden = out.last_hidden_state[:, -1, :]
#             logits = model.lm_head(last_hidden) # [1, Vocab]
            
#             # === Debug: 打印前5个概率最大的 token ===
#             if step == 0:
#                 top5_probs, top5_ids = torch.topk(torch.softmax(logits, dim=-1), k=5)
#                 print("Step 0 Top-5 Predictions:")
#                 for prob, idx in zip(top5_probs[0], top5_ids[0]):
#                     token_str = tokenizer.decode([idx.item()])
#                     print(f"  TokenID: {idx.item()} | Prob: {prob:.4f} | String: '{token_str}'")

#             # 贪婪解码
#             next_token_id = torch.argmax(logits, dim=-1)
            
#             # === 强制生成逻辑 ===
#             # 如果在前 5 步就想停止 (EOS)，强制选第二高分的 token，看它还能说什么
#             if next_token_id.item() == tokenizer.eos_token_id and step < 5:
#                 # 把 EOS 的分数设为负无穷，重新选
#                 logits[:, tokenizer.eos_token_id] = -float('inf')
#                 next_token_id = torch.argmax(logits, dim=-1)
            
#             # 真正的停止条件
#             if next_token_id.item() == tokenizer.eos_token_id:
#                 print(f"[Stop at step {step} due to EOS]")
#                 break
            
#             generated_ids.append(next_token_id.item())
            
#             # 实时打印生成的 token (不换行)
#             token_str = tokenizer.decode([next_token_id.item()])
#             print(f"{token_str}", end="|", flush=True) # 用 | 分隔方便看清楚
            
#             # 下一步输入
#             next_token_emb = model.llama.embed_tokens(next_token_id.unsqueeze(0))
#             curr_embeds = torch.cat([curr_embeds, next_token_emb], dim=1)
            
#         print("\n\nFull Generated Text:")
#         text = tokenizer.decode(generated_ids, skip_special_tokens=True)
#         print(f"'{text}'")
#         print("="*40)

# def main():
#     args = parse_args()
    
#     print(f"Loading tokenizer: {args.base_model_name}")
#     tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     print(f"Loading model from: {args.model_path}")
    
#     config = AutoConfig.from_pretrained(args.model_path)
#     if config.vocab_size == 32000:
#         print("Adjusting config.vocab_size to 32001.")
#         config.vocab_size = 32001

#     model = RAGLlamaDecoupled.from_pretrained(args.model_path, config=config)
#     model.to(args.device)
#     model.eval()

#     # 测试案例
#     print("\nRunning Test Case...")
#     test_q = "What is the capital of France?"
#     test_chunks = [
#         "Berlin is the capital of Germany.", 
#         "Paris is the capital and most populous city of France.",
#         "The Eiffel Tower is located in Paris.",
#         "Tokyo is the capital of Japan."
#     ]
    
#     run_inference(model, tokenizer, test_q, test_chunks, args.device, topk=args.topk_chunks)

# if __name__ == "__main__":
#     main()



# inference_decoupled.py

# import os
# import argparse
# from typing import List, Tuple

# import torch
# from transformers import AutoTokenizer, AutoConfig

# from rag_llama_decoupled import RAGLlamaDecoupled


# # ----------------- 参数 -----------------
# def parse_args():
#     parser = argparse.ArgumentParser(description="Inference for Decoupled RAG")
#     parser.add_argument(
#         "--model_path",
#         type=str,
        
#         default="/media/hc-sfxz/4738C1D329F4278F/zlt/version6/trained_decoupled/final_model_decoupled",
#         help="训练好的模型目录 (例如 ./trained_decoupled/final_model_decoupled)",
#     )
#     parser.add_argument(
#         "--base_model_name",
#         type=str,
#         default="/media/hc-sfxz/4738C1D329F4278F/zlt/version6/TinyLlama-1.1B-Chat-v1.0",
#         help="用于加载 tokenizer 的基础模型路径；如果不填则使用 model_path",
#     )
#     parser.add_argument(
#         "--device",
#         type=str,
#         default="cuda" if torch.cuda.is_available() else "cpu",
#     )
#     parser.add_argument("--max_new_tokens", type=int, default=64)
#     parser.add_argument("--topk_chunks", type=int, default=2, help="推理时保留的 chunk 数")
#     parser.add_argument(
#         "--max_ctx_len",
#         type=int,
#         default=1024,
#         help="和训练时保持一致的 context 最大 token 数",
#     )
#     parser.add_argument(
#         "--max_doc_len",
#         type=int,
#         default=512,
#         help="单个 doc 的最大 token 长度（训练中也是 512）",
#     )
#     return parser.parse_args()


# # ----------------- 上下文预处理（和训练脚本保持一致） -----------------
# def build_context_ids_and_spans(
#     tokenizer: AutoTokenizer,
#     chunks: List[str],
#     device,
#     max_ctx_len: int = 1024,
#     max_doc_len: int = 512,
# ) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[str]]:
#     """
#     和 RAGDecoupledDataset.__getitem__ 中的逻辑对齐：
#     - 所有 chunk 在 token 级按顺序拼成一条长序列
#     - 记录每个 chunk 在该长序列中的 (start, end)
#     - 用 pad_token_id 补到 max_ctx_len
#     """
#     pad_id = tokenizer.pad_token_id
#     all_ctx_ids: List[int] = []
#     chunk_spans: List[Tuple[int, int]] = []
#     kept_chunks: List[str] = []

#     for doc in chunks:
#         enc = tokenizer(
#             doc,
#             truncation=True,
#             max_length=max_doc_len,
#             add_special_tokens=False,
#             return_tensors="pt",
#         )
#         doc_ids = enc.input_ids.squeeze(0)  # [L_doc]
#         L_doc = doc_ids.size(0)
#         if L_doc == 0:
#             continue

#         # 超过 max_ctx_len 就停止追加后续 doc
#         if len(all_ctx_ids) + L_doc > max_ctx_len:
#             break

#         s = len(all_ctx_ids)
#         e = s + L_doc
#         all_ctx_ids.extend(doc_ids.tolist())
#         chunk_spans.append((s, e))
#         kept_chunks.append(doc)

#     if not all_ctx_ids:
#         all_ctx_ids = [pad_id]
#         chunk_spans = [(0, 1)]
#         kept_chunks = [""]

#     ctx_ids = torch.full(
#         (1, max_ctx_len),
#         pad_id,
#         dtype=torch.long,
#         device=device,
#     )
#     ctx_ids[0, : len(all_ctx_ids)] = torch.tensor(all_ctx_ids, dtype=torch.long, device=device)
#     return ctx_ids, chunk_spans, kept_chunks


# # ----------------- 单次推理 -----------------
# @torch.no_grad()
# def run_inference(
#     model: RAGLlamaDecoupled,
#     tokenizer: AutoTokenizer,
#     question: str,
#     chunks: List[str],
#     device,
#     topk: int = 2,
#     max_new_tokens: int = 64,
#     max_ctx_len: int = 1024,
#     max_doc_len: int = 512,
# ):
#     model.eval()

#     # 1. 编码 question（不加特殊 token，和训练对齐）
#     q_inputs = tokenizer(
#         question,
#         return_tensors="pt",
#         truncation=True,
#         max_length=512,
#         padding="max_length",
#         add_special_tokens=False,
#     ).to(device)
#     q_ids = q_inputs.input_ids  # [1, Q]

#     # 2. 编码 context，构造 context_ids & chunk_spans
#     context_ids, chunk_spans, used_chunks = build_context_ids_and_spans(
#         tokenizer, chunks, device, max_ctx_len=max_ctx_len, max_doc_len=max_doc_len
#     )
#     batch_chunk_spans = [chunk_spans]  # 单样本 batch

#     print("\n" + "=" * 60)
#     print(f"Question: {question}")
#     print(f"#Original Chunks: {len(chunks)}, #Used Chunks: {len(used_chunks)}")

#     # 3. 调用模型（answer_ids=None → 只做检索+融合，不做生成损失）
#     out = model(
#         question_ids=q_ids,
#         context_ids=context_ids,
#         answer_ids=None,
#         chunk_spans=batch_chunk_spans,
#         topk_chunks=topk,
#     )

#     prefix_emb = out["prefix_emb"]               # [1, M, H]
#     pruned_context_emb = out["pruned_context_emb"]  # [1, Kp, H]
#     q_emb_full = out["question_emb"]            # [1, Q, H]
#     chunk_scores_list = out["chunk_scores"]     # List[Tensor[J_b]]，这里只有一条
#     chunk_scores = chunk_scores_list[0]         # [J]

#     # 4. 打印检索结果
#     print("\n--- Retrieval (chunk scores) ---")
#     if chunk_scores.numel() == 0:
#         print("No valid chunks, fall back to raw context.")
#     else:
#         probs = torch.softmax(chunk_scores, dim=0)  # 转成归一化分数方便观察
#         J = chunk_scores.size(0)
#         k_eff = min(topk, J)
#         topk_vals, topk_idx = torch.topk(chunk_scores, k=k_eff)
#         topk_idx = topk_idx.tolist()
#         topk_set = set(topk_idx)

#         for i in range(J):
#             tag = ">>> SELECTED" if i in topk_set else "    ignored"
#             preview = used_chunks[i].replace("\n", " ")
#             if len(preview) > 80:
#                 preview = preview[:80] + "..."
#             print(f"{tag} | idx={i:02d} | score={probs[i]:.4f} | {preview}")

#     # 5. 生成答案：在 [prefix, pruned_ctx_emb, question_emb] 上做贪心解码
#     print("\n--- Generating Answer ---")
#     base_embeds = torch.cat([prefix_emb, pruned_context_emb, q_emb_full], dim=1)  # [1, L0, H]
#     curr_embeds = base_embeds
#     generated_ids: List[int] = []

#     for step in range(max_new_tokens):
#         B, L, H = curr_embeds.shape
#         attn_mask = torch.ones(B, L, dtype=torch.long, device=device)

#         llama_out = model.llama(
#             inputs_embeds=curr_embeds,
#             attention_mask=attn_mask,
#             return_dict=False,
#         )
#         hidden = llama_out[0]          # [1, L, H]
#         last_hidden = hidden[:, -1, :]  # [1, H]
#         logits = model.lm_head(last_hidden)  # [1, V]

#         next_token_id = torch.argmax(logits, dim=-1)  # [1]
#         token_id = next_token_id.item()

#         # 结束条件
#         if token_id == tokenizer.eos_token_id:
#             break

#         generated_ids.append(token_id)

#         # 将新 token 的 embedding 接到序列末尾
#         next_emb = model.llama.embed_tokens(next_token_id.unsqueeze(0))  # [1,1,H]
#         curr_embeds = torch.cat([curr_embeds, next_emb], dim=1)

#     if generated_ids:
#         answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
#     else:
#         answer = "(empty output)"

#     print(f"\nAnswer: {answer}")
#     print("=" * 60)
#     return answer, chunk_scores


# # ----------------- main -----------------
# def main():
#     args = parse_args()
#     device = torch.device(args.device)

#     # 1. 加载 tokenizer
#     tok_path = args.base_model_name or args.model_path
#     print(f"Loading tokenizer from: {tok_path}")
#     tokenizer = AutoTokenizer.from_pretrained(tok_path)
#     if tokenizer.pad_token_id is None:
#         # 训练时你显式添加了 pad_token，这里也要保证存在
#         tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

#     # 2. 加载模型
#     print(f"Loading model from: {args.model_path}")
#     model = RAGLlamaDecoupled.from_pretrained(args.model_path)
#     model.to(device)
#     model.eval()

#     # 3. 一个简单的测试样例
#     demo_q = "What is the capital of France?"
#     demo_chunks = [
#         "Berlin is the capital of Germany.",
#         "Paris is the capital and most populous city of France.",
#         "The Eiffel Tower is located in Paris.",
#         "Tokyo is the capital of Japan.",
#     ]
#     run_inference(
#         model,
#         tokenizer,
#         demo_q,
#         demo_chunks,
#         device=device,
#         topk=args.topk_chunks,
#         max_new_tokens=args.max_new_tokens,
#         max_ctx_len=args.max_ctx_len,
#         max_doc_len=args.max_doc_len,
#     )

#     # 4. 交互式模式
#     while True:
#         try:
#             print("\n\n--- Interactive Mode (Ctrl+C 退出) ---")
#             q = input("Input Question: ").strip()
#             if not q:
#                 continue

#             print("Input Context Chunks (每行一个，输入空行+END 结束):")
#             chunks = []
#             while True:
#                 line = input("> ")
#                 if line.strip() == "END":
#                     break
#                 if line.strip():
#                     chunks.append(line.strip())

#             if not chunks:
#                 print("No chunks provided, skip.")
#                 continue

#             run_inference(
#                 model,
#                 tokenizer,
#                 q,
#                 chunks,
#                 device=device,
#                 topk=args.topk_chunks,
#                 max_new_tokens=args.max_new_tokens,
#                 max_ctx_len=args.max_ctx_len,
#                 max_doc_len=args.max_doc_len,
#             )

#         except KeyboardInterrupt:
#             print("\nBye.")
#             break
#         except Exception as e:
#             print(f"Error: {e}")
#             import traceback

#             traceback.print_exc()


# if __name__ == "__main__":
#     main()



# inference_decoupled_v3.py
# ------------------------------------------------------------
# A: top-p + repetition penalty 采样，避免 nananana / The The
# B: 注入 "The answer is" 作为答案起始 prompt
# C: 打印 step0 top-10 token，debug 退化原因
# ------------------------------------------------------------

import argparse
import json
from typing import List, Tuple

import torch
from transformers import AutoTokenizer
from rag_llama_decoupled_latent import RAGLlamaDecoupledLatent


# =========================
# Args
# =========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=str, default = "/media/hc-sfxz/4738C1D329F4278F/zlt/version6/trained_decoupled/final_model_decoupled_1")
    p.add_argument("--llama_path", type=str, default = "/media/hc-sfxz/4738C1D329F4278F/zlt/version6/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # architecture (MUST match training)
    p.add_argument("--num_da_layers", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--prompt_max_len", type=int, default=32)
    p.add_argument("--enable_global", action="store_true")

    # inference
    p.add_argument("--topk_chunks", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--rep_penalty", type=float, default=1.15)

    p.add_argument("--debug_top10", action="store_true")
    p.add_argument("--no_answer_prompt", action="store_true")
    return p.parse_args()


# =========================
# Context builder (same as training)
# =========================
def build_context(
    tokenizer,
    docs: List[str],
    device,
    max_ctx_len=1024,
    max_doc_len=512,
):
    pad = tokenizer.pad_token_id
    all_ids = []
    spans = []
    kept_docs = []

    for doc in docs:
        ids = tokenizer(
            doc,
            truncation=True,
            max_length=max_doc_len,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        if ids.numel() == 0:
            continue
        if len(all_ids) + ids.numel() > max_ctx_len:
            break
        s = len(all_ids)
        e = s + ids.numel()
        all_ids.extend(ids.tolist())
        spans.append((s, e))
        kept_docs.append(doc)

    if not all_ids:
        all_ids = [pad]
        spans = [(0, 1)]
        kept_docs = [""]

    ctx_ids = torch.full((1, max_ctx_len), pad, dtype=torch.long, device=device)
    ctx_ids[0, :len(all_ids)] = torch.tensor(all_ids, dtype=torch.long, device=device)
    ctx_mask = torch.zeros((1, max_ctx_len), dtype=torch.long, device=device)
    ctx_mask[0, :len(all_ids)] = 1

    return ctx_ids, ctx_mask, spans, kept_docs


# =========================
# Sampling (A)
# =========================
def sample_next_token(
    logits,
    generated_ids,
    temperature=0.7,
    top_p=0.9,
    rep_penalty=1.15,
):
    logits = logits.clone()

    # repetition penalty
    if generated_ids:
        for tid in set(generated_ids):
            logits[:, tid] /= rep_penalty

    if temperature > 0:
        logits = logits / temperature

    probs = torch.softmax(logits, dim=-1)

    # nucleus top-p
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    mask = cum > top_p
    mask[:, 0] = False
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-9)

    next_idx = torch.multinomial(sorted_probs, 1)
    next_token = sorted_idx.gather(-1, next_idx).squeeze(1)
    return next_token


# =========================
# KV-cache generation
# =========================
@torch.no_grad()
def generate(
    model,
    tokenizer,
    base_embeds,
    base_mask,
    args,
):
    device = base_embeds.device
    B, L0, _ = base_embeds.shape

    pos0 = torch.arange(L0, device=device).unsqueeze(0)
    out0 = model.llama(
        inputs_embeds=base_embeds,
        attention_mask=base_mask,
        position_ids=pos0,
        use_cache=True,
        return_dict=True,
    )
    past = out0.past_key_values
    last = out0.last_hidden_state[:, -1, :]
    logits = model.lm_head(last)

    if args.debug_top10:
        p, idx = torch.topk(torch.softmax(logits, dim=-1), k=10)
        print("\n[DEBUG] Step0 Top-10 tokens:")
        for prob, tid in zip(p[0], idx[0]):
            print(f"  {tid.item():5d}  {prob.item():.4f}  {repr(tokenizer.decode([tid.item()]))}")

    generated = []
    next_id = sample_next_token(
        logits,
        generated,
        temperature=args.temperature,
        top_p=args.top_p,
        rep_penalty=args.rep_penalty,
    )

    if next_id.item() == tokenizer.eos_token_id:
        return ""

    generated.append(next_id.item())
    next_emb = model.llama.embed_tokens(next_id.unsqueeze(1))
    cur_mask = torch.cat([base_mask, torch.ones((B, 1), device=device, dtype=torch.long)], dim=1)
    cur_len = L0 + 1

    for _ in range(args.max_new_tokens - 1):
        pos = torch.tensor([[cur_len - 1]], device=device)
        out = model.llama(
            inputs_embeds=next_emb,
            attention_mask=cur_mask,
            position_ids=pos,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        logits = model.lm_head(out.last_hidden_state[:, -1, :])
        next_id = sample_next_token(
            logits,
            generated,
            temperature=args.temperature,
            top_p=args.top_p,
            rep_penalty=args.rep_penalty,
        )
        if next_id.item() == tokenizer.eos_token_id:
            break
        generated.append(next_id.item())
        next_emb = model.llama.embed_tokens(next_id.unsqueeze(1))
        cur_len += 1
        cur_mask = torch.cat([cur_mask, torch.ones((B, 1), device=device, dtype=torch.long)], dim=1)

    return tokenizer.decode(generated, skip_special_tokens=True)


# =========================
# Main inference
# =========================
@torch.no_grad()
def run_sample(model, tokenizer, sample, args):
    device = next(model.parameters()).device

    question = sample["question"]
    docs = sample["docs"]
    labels = sample.get("doc_labels", None)

    q_ids = tokenizer(
        question,
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        return_tensors="pt",
    ).input_ids.to(device)
    q_mask = (q_ids != tokenizer.pad_token_id).long()

    ctx_ids, ctx_mask, spans, kept_docs = build_context(tokenizer, docs, device)

    out = model(
        question_ids=q_ids,
        context_ids=ctx_ids,
        answer_ids=None,
        chunk_spans=[spans],
        chunk_labels=None,
        topk_chunks=args.topk_chunks,
    )

    scores = out["chunk_scores"][0]
    probs = torch.softmax(scores, dim=0)

    print("\n" + "=" * 90)
    print("Q:", question)
    print("\n--- Retrieval ---")

    topk = torch.topk(scores, k=min(args.topk_chunks, scores.numel())).indices.tolist()
    topk_set = set(topk)

    for i, (doc, p) in enumerate(zip(kept_docs, probs)):
        tag = ">>> SELECTED" if i in topk_set else "    ignored"
        gold = f" gold={labels[i]}" if labels is not None else ""
        preview = doc.replace("\n", " ")
        print(f"{tag} idx={i:02d} p={p.item():.4f}{gold} | {preview[:200]}")

    # prefix + pruned ctx + question
    prefix = out["prefix_emb"]
    pruned_ctx = out["pruned_context_emb"]
    q_emb = out["question_emb"]

    prefix_mask = torch.ones((1, prefix.size(1)), device=device, dtype=torch.long)
    pruned_mask = torch.ones((1, pruned_ctx.size(1)), device=device, dtype=torch.long)

    base_embeds = torch.cat([prefix, pruned_ctx, q_emb], dim=1)
    base_mask = torch.cat([prefix_mask, pruned_mask, q_mask], dim=1)

    # (B) answer prompt
    if not args.no_answer_prompt:
        prompt_ids = tokenizer(
            "The answer is",
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(device)
        prompt_emb = model.llama.embed_tokens(prompt_ids)
        prompt_mask = torch.ones((1, prompt_emb.size(1)), device=device, dtype=torch.long)
        base_embeds = torch.cat([base_embeds, prompt_emb], dim=1)
        base_mask = torch.cat([base_mask, prompt_mask], dim=1)

    ans = generate(model, tokenizer, base_embeds, base_mask, args)

    print("\n--- Answer ---")
    print(ans)
    print("Gold:", sample.get("answer"))
    print("=" * 90)


# =========================
# Entry
# =========================
def main():
    args = parse_args()
    device = torch.device(args.device)

    # IMPORTANT: load tokenizer from ckpt
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir)
    assert tokenizer.pad_token_id is not None

    model = RAGLlamaDecoupledLatent.from_pretrained(
        args.ckpt_dir,
        llama_model_name_or_path=args.llama_path,
        num_da_layers=args.num_da_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        prompt_max_len=args.prompt_max_len,
        enable_global=args.enable_global,
    ).to(device)

    model.llama.resize_token_embeddings(len(tokenizer))
    model.pad_id = tokenizer.pad_token_id
    model.eos_id = tokenizer.eos_token_id
    model.llama.config.use_cache = True

    # ===== your real sample =====
    sample = {
        "id": "train_0",
        "question": "Are director of film Move (1970 Film) and director of film Méditerranée (1963 Film) from the same country?",
        "answer": "The answer is no.",
        "original_answer": "no",
        "docs": [
            "[Stuart Rosenberg] Stuart Rosenberg (August 11, 1927 – March 15, 2007) was an American film and television director whose motion pictures include \"Cool Hand Luke\" (1967), \"Voyage of the Damned\" (1976), \"The Amityville Horror\" (1979), and \"The Pope of Greenwich Village\" (1984). He was noted for his work with actor Paul Newman.",
            "[Méditerranée (1963 film)] Méditerranée is a 1963 French experimental film directed by Jean-Daniel Pollet with assistance from Volker Schlöndorff. It was written by Philippe Sollers and produced by Barbet Schroeder, with music by Antione Duhamel. The 45 minute film is cited as one of Pollet's most influential films, which according to Jonathan Rosenbaum directly influenced Jean-Luc Goddard's \"Contempt\", released later the same year. Footage for the film was shot around the Mediterranean, including at a Greek temple, a Sicilian garden, the sea, and also features a fisherman, a bullfighter, and a girl on an operating table.",
            "[Move (1970 film)] Move is a 1970 American comedy film starring Elliott Gould, Paula Prentiss and Geneviève Waïte, and directed by Stuart Rosenberg. The screenplay was written by Joel Lieber and Stanley Hart, adapted from a novel by Lieber.",
            "[Ian Barry (director)] Ian Barry is an Australian director of film and TV.",
            "[Peter Levin] Peter Levin is an American director of film, television and theatre.",
            "[Brian Johnson (special effects artist)] Brian Johnson( born 1939 or 1940) is a British designer and director of film and television special effects.",
            "[Rachel Feldman] Rachel Feldman( born August 22, 1954) is an American director of film and television and screenwriter of television films.",
            "[Hanro Smitsman] Hanro Smitsman, born in 1967 in Breda( Netherlands), is a writer and director of film and television.",
            "[Jean-Daniel Pollet] Jean-Daniel Pollet (1936–2004) was a French film director and screenwriter who was most active in the 1960s and 1970s. He was associated with two approaches to filmmaking: comedies which blended burlesque and melancholic elements, and poetic films based on texts by writers such as the French poet Francis Ponge.",
            "[Howard W. Koch] Howard Winchel Koch( April 11, 1916 – February 16, 2001) was an American producer and director of film and television."
        ],
        "doc_labels": [1, 1, 1, 0, 0, 0, 0, 0, 1, 0]
    }

    run_sample(model, tokenizer, sample, args)


if __name__ == "__main__":
    main()
