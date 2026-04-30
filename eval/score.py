"""Scoring utilities: accuracy + macro-F1 for sentiment predictions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

def _accuracy(gold: list[str], pred: list[str]) -> float:
    if not gold:
        return 0.0
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def _macro_f1(gold: list[str], pred: list[str]) -> float:
    labels = sorted(set(gold))
    if not labels:
        return 0.0
    f1s: list[float] = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def score_predictions(path: str | Path) -> dict[str, Any]:
    """
    Read a predictions JSONL file and return a metrics dict with:
      - sentiment_accuracy, sentiment_macro_f1
      - parse_error_rate
      - n_samples, method, paradigm, backbone, dataset
    """
    path = Path(path)
    gold_sents: list[str] = []
    pred_sents: list[str] = []
    n_parse_errors = 0
    method = dataset = paradigm = backbone = ""

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)

            method   = rec.get("method", method)
            dataset  = rec.get("dataset", dataset)
            paradigm = rec.get("paradigm", paradigm)
            backbone = rec.get("backbone", backbone)

            g_sent = rec["gold"]["sentiment"]
            p_sent = rec["pred"]["sentiment"]

            if not rec.get("parse_ok", True):
                n_parse_errors += 1
                p_sent = "__WRONG__"

            gold_sents.append(g_sent)
            pred_sents.append(p_sent)

    n = len(gold_sents)

    return {
        "method": method,
        "paradigm": paradigm,
        "backbone": backbone,
        "dataset": dataset,
        "n_samples": n,
        "sentiment_accuracy": round(_accuracy(gold_sents, pred_sents), 4),
        "sentiment_macro_f1": round(_macro_f1(gold_sents, pred_sents), 4),
        "parse_error_rate": round(n_parse_errors / n, 4) if n > 0 else 0.0,
    }


def write_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
