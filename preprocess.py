import json
import argparse


def build_chunk_labels(ctx_titles, supporting_titles):
    sup_set = set(supporting_titles)
    labels = []
    for t in ctx_titles:
        labels.append(1 if t in sup_set else 0)
    return labels


def preprocess_direction_a(input_path: str, output_path: str):
    """
    input_path: 原始数据文件（JSON Lines格式，每行一个JSON对象）
    output_path: 导出为 cleaned_train_direction_a.json（或.jsonl）
    """
    # 读取JSON Lines文件（逐行解析）
    raw = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            raw.append(json.loads(line))  # 每行一个JSON对象

    out = []
    for ex in raw:
        q = ex["question"]
        golden = ex.get("golden_answers", [])
        if not golden:
            # 没有答案就跳过
            continue
        answer = golden[0]

        meta = ex.get("metadata", {})
        ctx = meta.get("context", {})
        ctx_titles = ctx.get("title", [])
        ctx_contents = ctx.get("content", [])
        supp = meta.get("supporting_facts", {})
        supp_titles = supp.get("title", [])

        # 构造 docs: 每个 doc = "[title] sent0 sent1 ..."
        docs = []
        for t, sents in zip(ctx_titles, ctx_contents):
            doc = f"[{t}] " + " ".join(sents)
            docs.append(doc)

        doc_labels = build_chunk_labels(ctx_titles, supp_titles)

        out.append({
            "id": ex.get("id", ""),
            "question": q,
            "answer": answer,
            "docs": docs,
            "doc_labels": doc_labels,
        })
    pos_counts = []
    for ex in out:
        labs = ex["doc_labels"]
        pos_counts.append(sum(labs))

    print("样本总数:", len(pos_counts))
    print("至少有1个正doc的比例:", sum(c > 0 for c in pos_counts) / len(pos_counts))
    print("平均每个样本的正doc个数:", sum(pos_counts) / len(pos_counts))
    print("pos_counts 前20个:", pos_counts[:20])    

    # 写入输出文件（如果需要保持JSON Lines格式，也可以逐行写入）
    with open(output_path, "w", encoding="utf-8") as f:
        if output_path.endswith(".jsonl"):
            # 按JSON Lines格式写入（每行一个对象）
            for item in out:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
        else:
            # 按标准JSON数组写入
            json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(out)} examples to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True,
                        help="原始 Hotpot 风格数据路径")
    parser.add_argument("--output_path", type=str, default="cleaned_train_direction_a.json",
                        help="输出的训练数据路径")
    args = parser.parse_args()
    preprocess_direction_a(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
