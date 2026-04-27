"""
Convert UIT-VSFC CSV files to unified JSONL format.

UIT-VSFC CSV columns: sentence, sentiment, topic
  sentiment: 0 = negative, 1 = neutral, 2 = positive
  topic:     0 = lecturer, 1 = training_program, 2 = facility, 3 = others

Output schema (one JSON per line):
  {
    "id": "vsfc_<split>_<row_index>",
    "dataset": "UIT-VSFC",
    "language": "vi",
    "task": "ABSA",
    "text": "<sentence>",
    "gold": {
      "aspect": "lecturer" | "training_program" | "facility" | "others",
      "sentiment": "positive" | "negative" | "neutral"
    }
  }

Usage:
  python scripts/prepare_vsfc_lcf_bert.py \
      --data-dir data/raw/uit-vsfc \
      --out-dir  data/processed/lcf_bert
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SENTIMENT_MAP = {"0": "negative", "1": "neutral", "2": "positive"}
TOPIC_MAP     = {"0": "lecturer", "1": "training_program", "2": "facility", "3": "others"}


def parse_csv(csv_path: Path, split: str) -> list[dict]:
    records: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            text      = row.get("sentence", "").strip()
            sentiment = SENTIMENT_MAP.get(row.get("sentiment", "").strip(), "")
            topic     = TOPIC_MAP.get(row.get("topic", "").strip(), "")

            if not text or not sentiment or not topic:
                continue

            records.append({
                "id": f"vsfc_{split}_{idx}",
                "dataset":  "UIT-VSFC",
                "language": "vi",
                "task": "ABSA",
                "text": text,
                "gold": {
                    "aspect": topic,
                    "sentiment": sentiment,
                },
            })
    return records


def write_jsonl(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare UIT-VSFC JSONL splits.")
    p.add_argument("--data-dir", type=Path, default=Path("data/raw/uit-vsfc"))
    p.add_argument("--out-dir",  type=Path, default=Path("data/processed/lcf_bert"))
    args = p.parse_args()

    for split in ("train", "val", "test"):
        csv_path = args.data_dir / f"{split}.csv"
        if not csv_path.exists():
            print(f"[SKIP] {csv_path} not found")
            continue
        records = parse_csv(csv_path, split)
        write_jsonl(records, args.out_dir / f"vsfc_{split}.jsonl")


if __name__ == "__main__":
    main()
