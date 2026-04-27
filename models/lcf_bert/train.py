"""Training script for LCF-BERT (joint aspect-category + sentiment classification).

Usage:
  python models/lcf_bert/train.py --config configs/lcf_bert_en.yaml
  python models/lcf_bert/train.py --config configs/lcf_bert_vi.yaml

The script:
  1. Loads train/val JSONL from data/processed/
  2. Fine-tunes LCF-BERT with two cross-entropy losses (sentiment + aspect)
  3. Saves best checkpoint to checkpoints/<dataset>/lcf_bert_best.pt
  4. Prints accuracy and macro-F1 on the val set after each epoch
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import BertModel

# Allow running from repo root: python models/lcf_bert/train.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.lcf_bert.data_utils import ABSADatasetJSONL, Tokenizer4Bert
from models.lcf_bert.lcf_bert import LCF_BERT

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ── Label maps ────────────────────────────────────────────────────────────────

SENTIMENT_MAP = {"negative": 0, "neutral": 1, "positive": 2, "conflict": 3}

ASPECT_MAPS: dict[str, dict[str, int]] = {
    "SemEval-2014-Restaurant": {
        "food": 0, "service": 1, "price": 2, "ambience": 3,
        "anecdotes/miscellaneous": 4, "restaurant": 4,   # merge rare
    },
    "SemEval-2014-Laptop": {},   # built dynamically from training data
    "UIT-VSFC": {
        "lecturer": 0, "training_program": 1, "facility": 2, "others": 3,
    },
}


def _build_aspect_map_from_jsonl(train_path: Path) -> dict[str, int]:
    """Collect all aspect labels from a JSONL file (for Laptop dataset)."""
    labels: set[str] = set()
    with train_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = __import__("json").loads(line)
                labels.add(rec["gold"]["aspect"])
    return {lbl: idx for idx, lbl in enumerate(sorted(labels))}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Collate (list of dicts → batch tensors) ───────────────────────────────────

def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {
        k: torch.tensor(np.stack([b[k] for b in batch]), dtype=torch.long)
        for k in keys
    }


# ── Eval on a DataLoader ──────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: LCF_BERT, loader: DataLoader, device: torch.device):
    model.eval()
    all_sent_gold, all_sent_pred = [], []
    all_asp_gold, all_asp_pred   = [], []

    for batch in loader:
        inputs = [
            batch["concat_bert_indices"].to(device),
            batch["concat_segments_indices"].to(device),
            batch["text_local_indices"].to(device),
            batch["aspect_indices"].to(device),
        ]
        sent_logits, asp_logits = model(inputs)

        all_sent_gold.extend(batch["polarity"].tolist())
        all_sent_pred.extend(sent_logits.argmax(dim=-1).cpu().tolist())
        all_asp_gold.extend(batch["aspect_label"].tolist())
        all_asp_pred.extend(asp_logits.argmax(dim=-1).cpu().tolist())

    sent_acc = accuracy_score(all_sent_gold, all_sent_pred)
    sent_f1  = f1_score(all_sent_gold, all_sent_pred, average="macro", zero_division=0)
    asp_acc  = accuracy_score(all_asp_gold, all_asp_pred)
    asp_f1   = f1_score(all_asp_gold, all_asp_pred, average="macro", zero_division=0)

    return {
        "sentiment_accuracy": sent_acc, "sentiment_macro_f1": sent_f1,
        "aspect_accuracy":    asp_acc,  "aspect_macro_f1":    asp_f1,
    }


# ── Main training loop ────────────────────────────────────────────────────────

def train(cfg: SimpleNamespace) -> None:
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Label maps ---
    aspect_map = ASPECT_MAPS.get(cfg.dataset, {})
    if not aspect_map:
        # Build from training data (e.g., Laptop)
        aspect_map = _build_aspect_map_from_jsonl(Path(cfg.train_file))
        logger.info("Built aspect map (%d labels) from training data", len(aspect_map))

    n_sentiments = len(SENTIMENT_MAP)
    n_aspects    = len(aspect_map)
    logger.info("Sentiments: %d  |  Aspects: %d", n_sentiments, n_aspects)

    # --- Tokeniser ---
    tokenizer = Tokenizer4Bert(cfg.max_seq_len, cfg.pretrained_bert_name)

    # --- Datasets ---
    train_ds = ABSADatasetJSONL(
        cfg.train_file, tokenizer, SENTIMENT_MAP, aspect_map
    )
    val_ds = ABSADatasetJSONL(
        cfg.val_file, tokenizer, SENTIMENT_MAP, aspect_map
    )
    logger.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True,  collate_fn=collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False, collate_fn=collate, num_workers=0)

    # --- Model ---
    opt = SimpleNamespace(
        bert_dim=cfg.bert_dim,
        dropout=cfg.dropout,
        max_seq_len=cfg.max_seq_len,
        local_context_focus=cfg.local_context_focus,
        SRD=cfg.SRD,
        polarities_dim=n_sentiments,
        aspects_dim=n_aspects,
        device=device,
    )
    bert = BertModel.from_pretrained(cfg.pretrained_bert_name)
    model = LCF_BERT(bert, opt).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.l2reg
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epoch
    )
    loss_fn = nn.CrossEntropyLoss()

    # --- Save path ---
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "lcf_bert_best.pt"

    best_val_acc = 0.0

    for epoch in range(1, cfg.num_epoch + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            inputs = [
                batch["concat_bert_indices"].to(device),
                batch["concat_segments_indices"].to(device),
                batch["text_local_indices"].to(device),
                batch["aspect_indices"].to(device),
            ]
            sent_labels = batch["polarity"].to(device)
            asp_labels  = batch["aspect_label"].to(device)

            sent_logits, asp_logits = model(inputs)
            loss = loss_fn(sent_logits, sent_labels) + loss_fn(asp_logits, asp_labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            total_loss += loss.item()

            if step % 50 == 0:
                logger.info("  epoch %d step %d/%d  loss=%.4f",
                            epoch, step, len(train_loader),
                            total_loss / step)

        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        logger.info(
            "Epoch %d/%d — sent_acc=%.4f sent_f1=%.4f  asp_acc=%.4f asp_f1=%.4f",
            epoch, cfg.num_epoch,
            metrics["sentiment_accuracy"], metrics["sentiment_macro_f1"],
            metrics["aspect_accuracy"],    metrics["aspect_macro_f1"],
        )

        if metrics["sentiment_accuracy"] > best_val_acc:
            best_val_acc = metrics["sentiment_accuracy"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "opt": vars(opt),
                "sentiment_map": SENTIMENT_MAP,
                "aspect_map": aspect_map,
                "val_metrics": metrics,
            }, best_path)
            logger.info("  ✔ Saved best checkpoint → %s", best_path)

    logger.info("Training done. Best val sentiment acc: %.4f", best_val_acc)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="Path to YAML config file, e.g. configs/lcf_bert_en.yaml")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    cfg = SimpleNamespace(**raw)
    train(cfg)


if __name__ == "__main__":
    main()
