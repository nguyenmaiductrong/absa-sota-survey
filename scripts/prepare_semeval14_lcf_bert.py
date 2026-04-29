"""
Convert SemEval-2014 XML files to unified JSONL format.

Output schema (one JSON per line):
  {
    "id": "<sentence_id>#<category>",
    "dataset": "SemEval-2014-Restaurant" | "SemEval-2014-Laptop",
    "language": "en",
    "task": "ABSA",
    "text": "<sentence text>",
    "gold": {
      "aspect": "<category>",  # food / service / price / ambience / ...
      "sentiment": "positive" | "negative" | "neutral"
    }
  }

One record per (sentence, aspect-category) pair.
Sentences with no <aspectCategories> block are skipped.
Aspects labelled `conflict` are dropped to match the standard 3-class ABSC
benchmark used by the LCF-BERT paper (Zeng et al., 2019, Table 2).

Usage:
  python scripts/prepare_semeval14_lcf_bert.py \
      --restaurant-train data/raw/semeval14/Restaurants_Train_v2.xml \
      --restaurant-test  data/raw/semeval14/Restaurants_Test_Gold.xml \
      --laptop-train     data/raw/semeval14/Laptop_Train_v2.xml \
      --laptop-test      data/raw/semeval14/Laptops_Test_Gold.xml \
      --out-dir          data/processed/lcf_bert
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


POLARITY_MAP = {
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
}


def parse_xml(xml_path: Path, dataset_name: str) -> list[dict]:
    """Parse one XML file into records.

    Prefer <aspectCategories> (Restaurant). Fall back to <aspectTerms>
    (Laptop) when the former is absent.
    """
    records: list[dict] = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for sentence in root.iter("sentence"):
        sid = sentence.get("id", "")
        text_el = sentence.find("text")
        if text_el is None or not text_el.text:
            continue
        text = text_el.text.strip()

        cats_el = sentence.find("aspectCategories")
        if cats_el is not None:
            for cat_el in cats_el.findall("aspectCategory"):
                aspect = cat_el.get("category", "").strip().lower()
                polarity = POLARITY_MAP.get(
                    cat_el.get("polarity", "").strip().lower(), "")
                if not aspect or not polarity:
                    continue
                records.append(_make_record(sid, dataset_name, text,
                                            aspect, polarity, "cat"))
            continue

        terms_el = sentence.find("aspectTerms")
        if terms_el is None:
            continue
        for term_el in terms_el.findall("aspectTerm"):
            aspect   = term_el.get("term", "").strip().lower()
            polarity = POLARITY_MAP.get(
                term_el.get("polarity", "").strip().lower(), "")
            if not aspect or not polarity:
                continue
            records.append(_make_record(sid, dataset_name, text,
                                        aspect, polarity, "term"))

    return records


def _make_record(sid: str, dataset: str, text: str,
                 aspect: str, sentiment: str, suffix: str) -> dict:
    return {
        "id": f"{sid}#{suffix}#{aspect}",
        "dataset": dataset,
        "language": "en",
        "task": "ABSA",
        "text": text,
        "gold": {
            "aspect": aspect,
            "sentiment": sentiment,
        },
    }


def write_jsonl(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare SemEval-2014 JSONL splits.")
    p.add_argument("--restaurant-train", type=Path,
                   default=Path("data/raw/semeval14/Restaurants_Train_v2.xml"))
    p.add_argument("--restaurant-test",  type=Path,
                   default=Path("data/raw/semeval14/Restaurants_Test_Gold.xml"))
    p.add_argument("--laptop-train",     type=Path,
                   default=Path("data/raw/semeval14/Laptop_Train_v2.xml"))
    p.add_argument("--laptop-test",      type=Path,
                   default=Path("data/raw/semeval14/Laptops_Test_Gold.xml"))
    p.add_argument("--out-dir",          type=Path, default=Path("data/processed/lcf_bert"))
    args = p.parse_args()

    splits = [
        (args.restaurant_train, "SemEval-2014-Restaurant", "semeval14_rest_train.jsonl"),
        (args.restaurant_test,  "SemEval-2014-Restaurant", "semeval14_rest_test.jsonl"),
        (args.laptop_train,     "SemEval-2014-Laptop",     "semeval14_lap_train.jsonl"),
        (args.laptop_test,      "SemEval-2014-Laptop",     "semeval14_lap_test.jsonl"),
    ]

    for xml_path, dataset_name, out_name in splits:
        if not xml_path.exists():
            print(f"[SKIP] {xml_path} not found")
            continue
        records = parse_xml(xml_path, dataset_name)
        write_jsonl(records, args.out_dir / out_name)


if __name__ == "__main__":
    main()
