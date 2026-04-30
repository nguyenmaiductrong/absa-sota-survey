"""
Tổng hợp + so sánh metrics InstructABSA với LCF-BERT (đã có sẵn).

Đọc results/metrics/{instruct_absa,lcf_bert}_*.json rồi in 1 bảng Markdown
súc tích cho báo cáo.

Usage:
  python scripts/summarize_instruct_absa_results.py
  python scripts/summarize_instruct_absa_results.py --results-dir results --md-out plans/reports/instruct_absa_results.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASET_ORDER = ["semeval14_rest", "semeval14_lap", "vsfc"]
DATASET_LABEL = {
    "semeval14_rest": "SemEval-14 Restaurant (en)",
    "semeval14_lap":  "SemEval-14 Laptop (en)",
    "vsfc":           "UIT-VSFC (vi)",
}
METHODS = ["lcf_bert", "instructabsa"]
METHOD_LABEL = {"lcf_bert": "LCF-BERT", "instructabsa": "InstructABSA"}


def load_metric(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def fmt(v, spec: str = ".4f") -> str:
    return "—" if v is None else format(v, spec)


def build_table(results_dir: Path) -> str:
    lines: list[str] = [
        "| Dataset | Method | Backbone | sent_acc | macro_F1 | latency (ms) | parse_err |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for ds in DATASET_ORDER:
        for m in METHODS:
            mp = results_dir / "metrics" / f"{m}_{ds}.json"
            data = load_metric(mp)
            if data is None:
                lines.append(f"| {DATASET_LABEL[ds]} | {METHOD_LABEL[m]} | (missing) | — | — | — | — |")
                continue
            lines.append(
                f"| {DATASET_LABEL[ds]} "
                f"| {data.get('method', METHOD_LABEL[m])} "
                f"| {data.get('backbone', '?')} "
                f"| {fmt(data.get('sentiment_accuracy'))} "
                f"| {fmt(data.get('sentiment_macro_f1'))} "
                f"| {fmt(data.get('avg_latency_ms'), '.2f')} "
                f"| {fmt(data.get('parse_error_rate'), '.3f')} |"
            )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--md-out",      type=Path, default=None,
                   help="Nếu cho, ghi bảng Markdown ra file (vd: plans/reports/...).")
    args = p.parse_args()

    table = build_table(args.results_dir)
    print(table)

    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        header = "# InstructABSA vs LCF-BERT — kết quả thực nghiệm\n\n"
        args.md_out.write_text(header + table + "\n", encoding="utf-8")
        print(f"\n[wrote] {args.md_out}")


if __name__ == "__main__":
    main()
