"""
Training entry point cho InstructABSA — fine-tune Seq2SeqLM với HF Trainer.

Usage:
  python -m models.instruct_absa.train --config configs/instruct_absa_en_restaurant.yaml

Config YAML cần các key (xem configs/instruct_absa_*.yaml):
  pretrained_model: str            # vd: "allenai/tk-instruct-base-def-pos"
  language:         "en" | "vi"
  train_jsonl:      path
  eval_jsonl:       path | null    # nếu null → dùng 10% train làm eval
  output_dir:       path           # vd: "checkpoints/semeval14_rest/instruct_absa_best"
  subsample_ratio:  float          # 0.5 = dùng 50% train (paper sample-eff)
  max_input_length: int            # 512
  max_target_length:int            # 8
  num_train_epochs: int            # 4
  batch_size:       int            # 4 (EN) / 2 (VI)
  grad_accum:       int            # 4
  learning_rate:    float          # 3e-5 / 5e-5
  weight_decay:     float          # 0.01
  warmup_ratio:     float          # 0.1
  fp16:             bool           # True (T4 ok)
  early_stop_patience: int         # 2
  seed:             int            # 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

# Cho phép chạy như script lẫn `python -m`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.instruct_absa.data_loader import load_split, read_jsonl, subsample, InstructABSADataset


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def split_train_eval(records: list[dict],
                     eval_ratio: float = 0.1,
                     seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Tách dev set khi config không cung cấp eval_jsonl."""
    import random
    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    n_eval = max(1, int(len(records) * eval_ratio))
    eval_idx = set(idx[:n_eval])
    train = [records[i] for i in idx[n_eval:]]
    evals = [records[i] for i in idx[:n_eval]]
    return train, evals


def main() -> None:
    parser = argparse.ArgumentParser(description="Train InstructABSA (ATSC).")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    pretrained = cfg["pretrained_model"]
    language   = cfg["language"]
    out_dir    = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] backbone={pretrained}  lang={language}  out={out_dir}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model     = AutoModelForSeq2SeqLM.from_pretrained(pretrained)

    # ---- Load data
    train_records = read_jsonl(cfg["train_jsonl"])
    train_records = subsample(train_records, cfg["subsample_ratio"], seed=cfg["seed"])

    if cfg.get("eval_jsonl"):
        eval_records = read_jsonl(cfg["eval_jsonl"])
    else:
        train_records, eval_records = split_train_eval(train_records, 0.1, seed=cfg["seed"])

    print(f"[train] train={len(train_records)}  eval={len(eval_records)}")

    train_ds = InstructABSADataset(train_records, tokenizer, language,
                                   cfg["max_input_length"], cfg["max_target_length"])
    eval_ds  = InstructABSADataset(eval_records, tokenizer, language,
                                   cfg["max_input_length"], cfg["max_target_length"])

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    # ---- Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        fp16=cfg["fp16"] and torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to=[],
        seed=cfg["seed"],
        predict_with_generate=False,
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=cfg["early_stop_patience"])]

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=callbacks,
    )

    trainer.train()

    # ---- Save best (load_best_model_at_end đã restore best vào trainer.model)
    final_dir = out_dir
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    metrics = trainer.evaluate()
    with (final_dir / "train_summary.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "pretrained": pretrained,
            "language": language,
            "n_train": len(train_records),
            "n_eval": len(eval_records),
            "subsample_ratio": cfg["subsample_ratio"],
            "final_eval": metrics,
        }, fh, ensure_ascii=False, indent=2)
    print(f"[train] saved best model -> {final_dir}")
    print(f"[train] final eval: {metrics}")


if __name__ == "__main__":
    main()
