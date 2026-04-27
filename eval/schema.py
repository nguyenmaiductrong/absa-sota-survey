"""Schema constants and prediction-file validator for the ABSA survey."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ── Allowed label sets ──────────────────────────────────────────────────────

ALLOWED_DATASETS = {
    "SemEval-2014-Restaurant",
    "SemEval-2014-Laptop",
    "UIT-VSFC",
}

ALLOWED_LANGUAGES = {"en", "vi"}

ALLOWED_TASKS = {
    "ABSA",         # joint aspect + sentiment (used in this survey)
    "ATSC",         # aspect-term sentiment only
    "ACSA",         # aspect-category sentiment only
}

ALLOWED_SENTIMENTS = {"positive", "negative", "neutral", "conflict"}

PARSE_ERROR_TOKEN = "__PARSE_ERROR__"


# ── Prediction-file validator ────────────────────────────────────────────────

def validate_predictions_file(path: str | Path) -> list[str]:
    """Return a list of error strings (empty = file is valid)."""
    errors: list[str] = []
    path = Path(path)

    if not path.exists():
        return [f"file not found: {path}"]

    required_top = {"id", "dataset", "language", "task", "text", "gold", "pred",
                    "raw_output", "parse_ok", "method", "paradigm", "backbone", "latency_ms"}

    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rec: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {lineno}: invalid JSON — {exc}")
                continue

            missing = required_top - rec.keys()
            if missing:
                errors.append(f"line {lineno}: missing fields {sorted(missing)}")
                continue

            # gold / pred sub-keys
            for block in ("gold", "pred"):
                for sub in ("aspect", "sentiment"):
                    if sub not in rec[block]:
                        errors.append(f"line {lineno}: missing {block}.{sub}")

            if rec["dataset"] not in ALLOWED_DATASETS:
                errors.append(f"line {lineno}: unknown dataset {rec['dataset']!r}")

            if rec["language"] not in ALLOWED_LANGUAGES:
                errors.append(f"line {lineno}: unknown language {rec['language']!r}")

            if rec["task"] not in ALLOWED_TASKS:
                errors.append(f"line {lineno}: unknown task {rec['task']!r}")

            parse_ok = rec.get("parse_ok")
            if not parse_ok:
                # Both pred fields must equal PARSE_ERROR_TOKEN
                for sub in ("aspect", "sentiment"):
                    val = rec.get("pred", {}).get(sub, "")
                    if val != PARSE_ERROR_TOKEN:
                        errors.append(
                            f"line {lineno}: parse_ok=false but pred.{sub}={val!r} "
                            f"(expected {PARSE_ERROR_TOKEN!r})"
                        )

            if not isinstance(rec.get("latency_ms"), (int, float)):
                errors.append(f"line {lineno}: latency_ms must be a number")

    return errors
