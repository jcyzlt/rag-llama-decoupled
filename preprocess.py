# import json
# import argparse


# def build_chunk_labels(ctx_titles, supporting_titles):
#     sup_set = set(supporting_titles)
#     labels = []
#     for t in ctx_titles:
#         labels.append(1 if t in sup_set else 0)
#     return labels


# def preprocess_direction_a(input_path: str, output_path: str):
#     """
#     input_path: 原始数据文件（JSON Lines格式，每行一个JSON对象）
#     output_path: 导出为 cleaned_train_direction_a.json（或.jsonl）
#     """
#     # 读取JSON Lines文件（逐行解析）
#     raw = []
#     with open(input_path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:  # 跳过空行
#                 continue
#             raw.append(json.loads(line))  # 每行一个JSON对象

#     out = []
#     for ex in raw:
#         q = ex["question"]
#         golden = ex.get("golden_answers", [])
#         if not golden:
#             # 没有答案就跳过
#             continue
#         answer = golden[0]

#         meta = ex.get("metadata", {})
#         ctx = meta.get("context", {})
#         ctx_titles = ctx.get("title", [])
#         ctx_contents = ctx.get("content", [])
#         supp = meta.get("supporting_facts", {})
#         supp_titles = supp.get("title", [])

#         # 构造 docs: 每个 doc = "[title] sent0 sent1 ..."
#         docs = []
#         for t, sents in zip(ctx_titles, ctx_contents):
#             doc = f"[{t}] " + " ".join(sents)
#             docs.append(doc)

#         doc_labels = build_chunk_labels(ctx_titles, supp_titles)

#         out.append({
#             "id": ex.get("id", ""),
#             "question": q,
#             "answer": answer,
#             "docs": docs,
#             "doc_labels": doc_labels,
#         })

#     # 写入输出文件（如果需要保持JSON Lines格式，也可以逐行写入）
#     with open(output_path, "w", encoding="utf-8") as f:
#         if output_path.endswith(".jsonl"):
#             # 按JSON Lines格式写入（每行一个对象）
#             for item in out:
#                 json.dump(item, f, ensure_ascii=False)
#                 f.write("\n")
#         else:
#             # 按标准JSON数组写入
#             json.dump(out, f, ensure_ascii=False, indent=2)

#     print(f"Saved {len(out)} examples to {output_path}")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_path", type=str, required=True,
#                         help="原始 Hotpot 风格数据路径")
#     parser.add_argument("--output_path", type=str, default="cleaned_train_direction_a.json",
#                         help="输出的训练数据路径")
#     args = parser.parse_args()
#     preprocess_direction_a(args.input_path, args.output_path)


# if __name__ == "__main__":
#     main()


# preprocess.py

# import json
# import argparse

# def build_chunk_labels(ctx_titles, supporting_titles):
#     """
#     根据 context 中的标题列表和 supporting_facts 中的标题列表，
#     构建二进制标签列表。
#     """
#     # 转为集合加速查找
#     sup_set = set(supporting_titles)
#     labels = []
    
#     for t in ctx_titles:
#         # 如果当前 chunk 的标题在 supporting_facts 里，就是正例
#         if t in sup_set:
#             labels.append(1)
#         else:
#             labels.append(0)
#     return labels

# def preprocess_direction_a(input_path: str, output_path: str):
#     """
#     input_path: 原始 HotpotQA 风格数据文件（JSON Lines格式）
#     output_path: 预处理后的 JSONL 文件
#     """
#     print(f"Reading from {input_path}...")
    
#     raw = []
#     with open(input_path, "r", encoding="utf-8") as f:
#         for line_num, line in enumerate(f):
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 raw.append(json.loads(line))
#             except json.JSONDecodeError as e:
#                 print(f"Error parsing line {line_num}: {e}")
#                 continue

#     out = []
#     skipped_no_answer = 0
#     skipped_no_context = 0

#     for ex in raw:
#         q = ex.get("question", "").strip()
#         golden = ex.get("golden_answers", [])
        
#         # 1. 检查是否有答案
#         if not golden:
#             skipped_no_answer += 1
#             continue
        
#         # 通常取第一个答案，并做简单清洗
#         answer = golden[0].strip()
        
#         meta = ex.get("metadata", {})
        
#         # 2. 检查是否有上下文
#         ctx = meta.get("context", {})
#         ctx_titles = ctx.get("title", [])     # 列表
#         ctx_contents = ctx.get("content", []) # 列表的列表
        
#         if not ctx_titles or not ctx_contents:
#             skipped_no_context += 1
#             continue
            
#         # 3. 获取支撑事实（Ground Truth）
#         supp = meta.get("supporting_facts", {})
#         supp_titles = supp.get("title", [])
        
#         # 4. 构造 docs 和 labels
#         docs = []
#         valid_titles = []
        
#         # 有时候 titles 和 contents 长度不一致，取最短
#         min_len = min(len(ctx_titles), len(ctx_contents))
        
#         for i in range(min_len):
#             t = ctx_titles[i]
#             sents = ctx_contents[i]
            
#             # 拼接: "[Title] sentence1. sentence2."
#             # 注意: sents 是一个句子列表，用空格连接
#             if isinstance(sents, list):
#                 doc_text = f"[{t}] " + " ".join(sents)
#             else:
#                 doc_text = f"[{t}] {sents}"
                
#             docs.append(doc_text)
#             valid_titles.append(t)

#         # 构建 Label
#         doc_labels = build_chunk_labels(valid_titles, supp_titles)
        
#         # 只有当 docs 不为空时才加入
#         if docs:
#             out.append({
#                 "id": ex.get("id", ""),
#                 "question": q,
#                 "answer": answer,
#                 "docs": docs,
#                 "doc_labels": doc_labels,
#             })

#     # 统计信息
#     total = len(out)
#     pos_counts = [sum(x["doc_labels"]) for x in out]
    
#     if total > 0:
#         has_pos_ratio = sum(1 for c in pos_counts if c > 0) / total
#         avg_pos = sum(pos_counts) / total
#     else:
#         has_pos_ratio = 0
#         avg_pos = 0

#     print("-" * 40)
#     print(f"Preprocess Done.")
#     print(f"Total raw examples: {len(raw)}")
#     print(f"Skipped (no answer): {skipped_no_answer}")
#     print(f"Skipped (no context): {skipped_no_context}")
#     print(f"Total valid examples saved: {total}")
#     print(f"Ratio of examples with at least 1 positive doc: {has_pos_ratio:.2%}")
#     print(f"Average positive docs per example: {avg_pos:.2f}")
#     print("-" * 40)

#     # 保存
#     with open(output_path, "w", encoding="utf-8") as f:
#         # 推荐保存为 jsonl 方便逐行读取
#         for item in out:
#             json.dump(item, f, ensure_ascii=False)
#             f.write("\n")

#     print(f"Saved to {output_path}")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_path", type=str, required=True, help="原始数据路径")
#     parser.add_argument("--output_path", type=str, default="cleaned_train_direction_a.jsonl", help="输出路径")
#     args = parser.parse_args()
    
#     preprocess_direction_a(args.input_path, args.output_path)

# if __name__ == "__main__":
#     main()



# import json
# import argparse
# import torch
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # ================= 配置区 =================
# # 定义 Few-Shot Prompt 模板，教 TinyLlama 怎么做事
# PROMPT_TEMPLATE = """You are a helpful assistant. Rewrite the Short Answer into a complete, natural sentence based on the Question.
# Do not add extra facts.

# Q: What is the capital of France?
# Short Answer: Paris
# Rewritten: The capital of France is Paris.

# Q: Who written "1984"?
# Short Answer: George Orwell
# Rewritten: "1984" was written by George Orwell.

# Q: Is Python compiled?
# Short Answer: no
# Rewritten: No, Python is not a compiled language.

# Q: {question}
# Short Answer: {short_answer}
# Rewritten:"""

# def load_local_model(model_path, device):
#     print(f"Loading TinyLlama from {model_path}...")
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
        
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path, 
#         torch_dtype=torch.float16, 
#         device_map=device
#     )
#     model.eval()
#     return tokenizer, model

# def rewrite_answer_with_model(model, tokenizer, question, short_answer, device):
#     # 构造输入
#     prompt = PROMPT_TEMPLATE.format(question=question, short_answer=short_answer)
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
#     # 生成
#     # max_new_tokens 不用太长，因为扩充后的句子通常也就一两句
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs, 
#             max_new_tokens=50, 
#             do_sample=False, # 使用贪婪解码保证稳定性
#             pad_token_id=tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id
#         )
    
#     # 解码
#     full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     # 提取 "Rewritten:" 之后的部分
#     # TinyLlama 可能会重复 prompt，所以要截取
#     # 简单解析逻辑：找到最后一个 "Rewritten:"
#     if "Rewritten:" in full_output:
#         # 取最后一个 split，防止 prompt 里自带 Rewritten 干扰
#         generated_part = full_output.split("Rewritten:")[-1].strip()
#         # 有时候模型会多嘴，取第一行
#         first_line = generated_part.split("\n")[0].strip()
#         return first_line
#     else:
#         # 如果生成失败，返回原答案，但在前面加 "The answer is " 兜底
#         return f"The answer is {short_answer}."

# def build_chunk_labels(ctx_titles, supporting_titles):
#     sup_set = set(supporting_titles)
#     labels = []
#     for t in ctx_titles:
#         if t in sup_set:
#             labels.append(1)
#         else:
#             labels.append(0)
#     return labels

# def preprocess_and_rewrite(input_path, output_path, model_path, device="cuda"):
#     # 1. 加载模型
#     tokenizer, model = load_local_model(model_path, device)
    
#     print(f"Reading raw data from {input_path}...")
#     raw = []
#     with open(input_path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 try:
#                     raw.append(json.loads(line))
#                 except: pass

#     out = []
#     skipped = 0
    
#     print("Processing and rewriting answers...")
#     # 使用 tqdm 显示进度
#     for ex in tqdm(raw):
#         q = ex.get("question", "").strip()
#         golden = ex.get("golden_answers", [])
#         if not golden:
#             skipped += 1
#             continue
            
#         short_answer = golden[0].strip()
        
#         # === 核心：调用模型改写答案 ===
#         # 策略：如果答案太短（< 15个词），就改写；否则保留
#         if len(short_answer.split()) < 15:
#             long_answer = rewrite_answer_with_model(model, tokenizer, q, short_answer, device)
#         else:
#             long_answer = short_answer
            
#         # 处理 Context 和 Labels (原有逻辑)
#         meta = ex.get("metadata", {})
#         ctx = meta.get("context", {})
#         ctx_titles = ctx.get("title", [])
#         ctx_contents = ctx.get("content", [])
#         supp_titles = meta.get("supporting_facts", {}).get("title", [])
        
#         if not ctx_titles:
#             skipped += 1
#             continue
            
#         docs = []
#         valid_titles = []
#         min_len = min(len(ctx_titles), len(ctx_contents))
        
#         for i in range(min_len):
#             t = ctx_titles[i]
#             sents = ctx_contents[i]
#             if isinstance(sents, list):
#                 doc_text = f"[{t}] " + " ".join(sents)
#             else:
#                 doc_text = f"[{t}] {sents}"
#             docs.append(doc_text)
#             valid_titles.append(t)
            
#         doc_labels = build_chunk_labels(valid_titles, supp_titles)
        
#         if docs:
#             out.append({
#                 "id": ex.get("id", ""),
#                 "question": q,
#                 "answer": long_answer, # 这里保存改写后的长答案
#                 "original_answer": short_answer, # 保留原答案备查
#                 "docs": docs,
#                 "doc_labels": doc_labels,
#             })
#         if len(out)>100:break
#     print(f"Processed {len(out)} examples. Skipped {skipped}.")
    
#     # 保存
#     with open(output_path, "w", encoding="utf-8") as f:
#         for item in out:
#             json.dump(item, f, ensure_ascii=False)
#             f.write("\n")
            
#     print(f"Saved to {output_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_path", type=str, required=True)
#     parser.add_argument("--output_path", type=str, default="cleaned_augmented.jsonl")
#     # 这里填你 TinyLlama 的本地路径
#     parser.add_argument("--model_path", type=str, default="/media/hc-sfxz/4738C1D329F4278F/zlt/version6/TinyLlama-1.1B-Chat-v1.0",help="Path to local TinyLlama model")
#     args = parser.parse_args()
    
#     preprocess_and_rewrite(
#         args.input_path, 
#         args.output_path, 
#         args.model_path,
#         device="cuda" if torch.cuda.is_available() else "cpu"
#     )


import json
import argparse
from tqdm import tqdm

def rewrite_rule_based(question, short_answer):
    """
    使用规则改写短答案，保证 100% 的准确率和鲁棒性。
    """
    short_answer = short_answer.strip()
    
    # 1. 处理 Yes/No
    lower_a = short_answer.lower()
    if lower_a in ["yes", "yes."]:
        return "The answer is yes."
    if lower_a in ["no", "no."]:
        return "The answer is no."
        
    # 2. 处理日期 (如果答案包含数字，大概率是日期或数量)
    # 简单模板: The answer is ...
    return f"The answer is {short_answer}."

def build_chunk_labels(ctx_titles, supporting_titles):
    sup_set = set(supporting_titles)
    labels = []
    for t in ctx_titles:
        if t in sup_set:
            labels.append(1)
        else:
            labels.append(0)
    return labels

def preprocess_with_rules(input_path, output_path):
    print(f"Reading raw data from {input_path}...")
    raw = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    raw.append(json.loads(line))
                except: pass

    out = []
    skipped = 0
    
    print("Processing and rewriting answers using RULES...")
    for ex in tqdm(raw):
        q = ex.get("question", "").strip()
        golden = ex.get("golden_answers", [])
        if not golden:
            skipped += 1
            continue
            
        short_answer = golden[0].strip()
        
        # === 核心：规则改写 ===
        # 只有当答案比较短的时候才加前缀，防止把长句子搞乱
        if len(short_answer.split()) < 10:
            long_answer = rewrite_rule_based(q, short_answer)
        else:
            long_answer = short_answer
            
        # 处理 Context
        meta = ex.get("metadata", {})
        ctx = meta.get("context", {})
        ctx_titles = ctx.get("title", [])
        ctx_contents = ctx.get("content", [])
        supp_titles = meta.get("supporting_facts", {}).get("title", [])
        
        if not ctx_titles:
            skipped += 1
            continue
            
        docs = []
        valid_titles = []
        min_len = min(len(ctx_titles), len(ctx_contents))
        
        for i in range(min_len):
            t = ctx_titles[i]
            sents = ctx_contents[i]
            if isinstance(sents, list):
                doc_text = f"[{t}] " + " ".join(sents)
            else:
                doc_text = f"[{t}] {sents}"
            docs.append(doc_text)
            valid_titles.append(t)
            
        doc_labels = build_chunk_labels(valid_titles, supp_titles)
        
        if docs:
            out.append({
                "id": ex.get("id", ""),
                "question": q,
                "answer": long_answer,
                "original_answer": short_answer,
                "docs": docs,
                "doc_labels": doc_labels,
            })

    print(f"Processed {len(out)} examples. Skipped {skipped}.")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in out:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
            
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="cleaned_augmented_rules.jsonl")
    args = parser.parse_args()
    
    preprocess_with_rules(args.input_path, args.output_path)
