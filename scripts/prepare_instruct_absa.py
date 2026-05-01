"""
Convert unified JSONL splits (đã có ở data/processed/lcf_bert/) → instruction
format cho InstructABSA (ATSC subtask, given-aspect).

Output schema (one JSON per line):
  {
    "id":          "<sentence_id>",
    "dataset":     "SemEval-2014-Restaurant" | "SemEval-2014-Laptop" | "UIT-VSFC",
    "language":    "en" | "vi",
    "task":        "ATSC",
    "raw_text":    "<text>",
    "aspect":      "<aspect/topic>",
    "sentiment":   "positive" | "negative" | "neutral",
    "input_text":  "<text> | <aspect>",
    "output_text": "positive" | "negative" | "neutral"
  }

`input_text` bám sát convention InstructABSA gốc (Scaria et al. 2024):
    "{sentence} | {aspect_term}"  →  "{polarity}"

Train script sẽ prepend instruction prompt vào `input_text` ở runtime.

Usage:
  python scripts/prepare_instruct_absa.py \
      --src-dir data/processed/lcf_bert \
      --out-dir data/processed/instruct_absa
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Map dataset → (split files đã có, slug output)
SPLIT_MAP = {
    "semeval14_rest": ("ATSC", "en"),
    "semeval14_lap":  ("ATSC", "en"),
    "vsfc":           ("ATSC", "vi"),
}

SPLIT_NAMES = {
    "semeval14_rest": ("train", "test"),
    "semeval14_lap":  ("train", "test"),
    "vsfc":           ("train", "val", "test"),
}


def convert_record(rec: dict, task: str) -> dict | None:
    text     = rec.get("text", "").strip()
    gold     = rec.get("gold", {}) or {}
    aspect   = gold.get("aspect", "").strip()
    polarity = gold.get("sentiment", "").strip().lower()

    if not text or not aspect or polarity not in ("positive", "negative", "neutral"):
        return None

    return {
        "id":          rec["id"],
        "dataset":     rec["dataset"],
        "language":    rec["language"],
        "task":        task,
        "raw_text":    text,
        "aspect":      aspect,
        "sentiment":   polarity,
        "input_text":  f"{text} | {aspect}",
        "output_text": polarity,
    }


def convert_split(src: Path, dst: Path, task: str) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out = convert_record(rec, task)
            if out is None:
                continue
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n:5d} records -> {dst}")
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare InstructABSA ATSC splits.")
    p.add_argument("--src-dir", type=Path, default=Path("data/processed/lcf_bert"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed/instruct_absa"))
    args = p.parse_args()

    total = 0
    for slug, (task, _lang) in SPLIT_MAP.items():
        for split in SPLIT_NAMES[slug]:
            src = args.src_dir / f"{slug}_{split}.jsonl"
            if not src.exists():
                print(f"[SKIP] {src} not found")
                continue
            dst = args.out_dir / f"{slug}_{split}.jsonl"
            total += convert_split(src, dst, task)

    print(f"Done. Total instruction-format records: {total}")


if __name__ == "__main__":
    main()
