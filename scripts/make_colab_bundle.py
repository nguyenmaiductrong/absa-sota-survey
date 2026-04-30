"""
Đóng gói các file CẦN THIẾT để upload lên Google Colab cho Phase 5
(train + evaluate InstructABSA). Không bundle: .git, raw XML/CSV, checkpoints
LCF-BERT, predictions cũ, node_modules, v.v.

Output: instruct_absa_bundle.zip ở thư mục gốc repo.

Usage:
  python scripts/make_colab_bundle.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ZIP   = REPO_ROOT / "instruct_absa_bundle.zip"

# Các đường dẫn (relative to repo root) cần đưa lên Colab
INCLUDE: list[str] = [
    # Eval framework
    "evaluate.py",
    "eval/__init__.py",
    "eval/schema.py",
    "eval/score.py",

    # InstructABSA module
    "models/__init__.py",
    "models/instruct_absa/__init__.py",
    "models/instruct_absa/instructions.py",
    "models/instruct_absa/data_loader.py",
    "models/instruct_absa/train.py",

    # Predictor (+ package init)
    "predictors/__init__.py",
    "predictors/instruct_absa.py",

    # Configs
    "configs/instruct_absa_en_restaurant.yaml",
    "configs/instruct_absa_en_laptop.yaml",
    "configs/instruct_absa_vi.yaml",

    # Train data (đã prep ở Phase 1)
    "data/processed/instruct_absa/semeval14_rest_train.jsonl",
    "data/processed/instruct_absa/semeval14_rest_test.jsonl",
    "data/processed/instruct_absa/semeval14_lap_train.jsonl",
    "data/processed/instruct_absa/semeval14_lap_test.jsonl",
    "data/processed/instruct_absa/vsfc_train.jsonl",
    "data/processed/instruct_absa/vsfc_val.jsonl",
    "data/processed/instruct_absa/vsfc_test.jsonl",

    # Test JSONL chuẩn cho evaluate.py
    "data/processed/lcf_bert/semeval14_rest_test.jsonl",
    "data/processed/lcf_bert/semeval14_lap_test.jsonl",
    "data/processed/lcf_bert/vsfc_test.jsonl",
]


def ensure_init(rel_path: str) -> None:
    """Tạo __init__.py rỗng nếu chưa có (cần cho Python package import)."""
    p = REPO_ROOT / rel_path
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")


def main() -> None:
    # Đảm bảo các __init__.py rỗng tồn tại (eval, models, predictors)
    for init in ("eval/__init__.py", "models/__init__.py", "predictors/__init__.py"):
        ensure_init(init)

    missing: list[str] = []
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in INCLUDE:
            src = REPO_ROOT / rel
            if not src.exists():
                missing.append(rel)
                continue
            zf.write(src, arcname=rel)
            print(f"  + {rel}")

    if missing:
        print("\n[WARN] missing files (skipped):")
        for m in missing:
            print(f"  - {m}")

    size_mb = OUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"\nWrote {OUT_ZIP} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
