"""
webMedQA → 健康知识库转换脚本 (康语 KangYu)
────────────────────────────────────────────
将 webMedQA 真实医患问答数据集转换为项目兼容的 health_qa.json 格式。

webMedQA 格式 (TSV):
  department  label  question_id  question  answer
  - department: 科室名称 (23个科室)
  - label: 1=采纳答案, 0=未采纳
  - question_id: 问题ID
  - question: 患者问题
  - answer: 医生回答

输出格式 (JSON):
  [{"question": "...", "answer": "...", "knowledge_point": "..."}, ...]

用法:
  python scripts/convert_webmedqa.py                    # 默认参数
  python scripts/convert_webmedqa.py --max 5000         # 限制最大条数
  python scripts/convert_webmedqa.py --min-answer-len 100  # 最小答案长度
"""
import os
import json
import argparse
import hashlib
from collections import Counter

# ── 科室名映射 ──────────────────────────────────────────────
# 将 webMedQA 的科室名映射为更友好的健康主题名
DEPARTMENT_MAP = {
    "内科": "内科常见病",
    "外科": "外科与创伤",
    "妇产科": "妇产科健康",
    "儿科": "儿科与儿童保健",
    "皮肤性病科": "皮肤与性病健康",
    "五官科": "眼耳鼻喉口腔健康",
    "肿瘤科": "肿瘤防治与康复",
    "心理健康科": "心理健康",
    "中医科": "中医养生",
    "传染科": "传染病预防",
    "整形美容科": "整形与美容",
    "美容": "皮肤与美容",
    "药品": "用药安全",
    "辅助检查科": "体检与检查",
    "保健养生": "日常保健养生",
    "康复医学科": "康复医学",
    "家居环境": "生活环境与健康",
    "子女教育": "儿童成长与教育",
    "运动瘦身": "科学运动与减重",
    "营养保健科": "营养膳食",
    "遗传": "遗传与优生",
    "体检科": "体检与筛查",
    "其他科室": "综合健康",
}


def parse_webmedqa(filepath: str) -> list[dict]:
    """解析 webMedQA TSV 文件，只保留 label=1 的采纳答案"""
    entries = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            department, label, qid, question, answer = parts[0], parts[1], parts[2], parts[3], parts[4]

            # 只保留采纳答案
            if label != "1":
                continue

            question = question.strip()
            answer = answer.strip()

            if not question or not answer:
                continue

            entries.append({
                "question": question,
                "answer": answer,
                "knowledge_point": DEPARTMENT_MAP.get(department, department),
                "department": department,
            })
    return entries


def clean_entries(entries: list[dict], min_answer_len: int = 80) -> list[dict]:
    """清洗：过滤短答案、去重"""
    # 按答案长度过滤
    filtered = [e for e in entries if len(e["answer"]) >= min_answer_len]

    # 按问题文本去重（保留第一次出现的 — 即更长的答案）
    seen_questions: set[str] = set()
    deduped = []
    for e in filtered:
        q_hash = hashlib.md5(e["question"].encode("utf-8")).hexdigest()
        if q_hash not in seen_questions:
            seen_questions.add(q_hash)
            deduped.append(e)

    return deduped


def sample_by_department(entries: list[dict], max_total: int | None = None,
                         max_per_department: int | None = None) -> list[dict]:
    """按科室均匀采样，保持多样性"""
    if max_total is None and max_per_department is None:
        return entries

    from collections import defaultdict
    by_dept: dict[str, list] = defaultdict(list)
    for e in entries:
        by_dept[e["knowledge_point"]].append(e)

    sampled = []
    per_dept = max_per_department or 0

    for dept, items in by_dept.items():
        if per_dept and len(items) > per_dept:
            # 均匀采样
            step = len(items) / per_dept
            selected = [items[int(i * step)] for i in range(per_dept)]
            sampled.extend(selected)
        else:
            sampled.extend(items)

    if max_total and len(sampled) > max_total:
        # 全局均匀采样
        step = len(sampled) / max_total
        sampled = [sampled[int(i * step)] for i in range(max_total)]

    return sampled


def main():
    parser = argparse.ArgumentParser(description="webMedQA → health_qa.json 转换")
    parser.add_argument("--input-dir", default="data/webMedQA",
                        help="webMedQA 数据目录")
    parser.add_argument("--output", default="data/Health-QA-Dataset/health_qa.json",
                        help="输出 JSON 路径")
    parser.add_argument("--min-answer-len", type=int, default=80,
                        help="最小答案长度（字符）默认 80")
    parser.add_argument("--max", type=int, default=None,
                        help="最大输出条数（None=不限制）")
    parser.add_argument("--max-per-department", type=int, default=500,
                        help="每科室最大条数（默认 500，保持多样性）")
    parser.add_argument("--files", nargs="*",
                        default=["medQA.test.txt", "medQA.valid.txt"],
                        help="要处理的文件列表")
    args = parser.parse_args()

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    all_entries = []
    for filename in args.files:
        filepath = os.path.join(ROOT, args.input_dir, filename)
        if not os.path.exists(filepath):
            print(f"⚠ 跳过不存在的文件: {filepath}")
            continue
        entries = parse_webmedqa(filepath)
        print(f"✓ {filename}: 解析到 {len(entries)} 条采纳答案")
        all_entries.extend(entries)

    print(f"\n总计采纳答案: {len(all_entries)} 条")

    # 清洗
    all_entries = clean_entries(all_entries, min_answer_len=args.min_answer_len)
    print(f"清洗后 (答案≥{args.min_answer_len}字 + 去重): {len(all_entries)} 条")

    # 采样
    all_entries = sample_by_department(
        all_entries,
        max_total=args.max,
        max_per_department=args.max_per_department,
    )
    print(f"采样后: {len(all_entries)} 条")

    # 统计
    kp_counts = Counter(e["knowledge_point"] for e in all_entries)
    print(f"\n科室分布 ({len(kp_counts)} 个科室):")
    for kp, count in kp_counts.most_common():
        print(f"  {kp}: {count} 条")

    # 输出为兼容格式（去掉内部字段）
    output = [
        {
            "question": e["question"],
            "answer": e["answer"],
            "knowledge_point": e["knowledge_point"],
        }
        for e in all_entries
    ]

    output_path = os.path.join(ROOT, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 备份旧文件
    if os.path.exists(output_path):
        backup = output_path + ".backup"
        print(f"\n📦 备份旧数据集 → {backup}")
        os.replace(output_path, backup)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完成！输出: {output_path}")
    print(f"   共 {len(output)} 条问答 / {len(kp_counts)} 个主题")


if __name__ == "__main__":
    main()
