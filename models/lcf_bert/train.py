"""Train LCF-BERT for the ABSA survey repo.

Run examples:
  python models/lcf_bert/train.py --config configs/lcf_bert_en_restaurant.yaml
  python models/lcf_bert/train.py --config configs/lcf_bert_vi.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModel

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


sentiment_map: dict[str, int] = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
    "conflict": 3,
}

# Closed aspect/category sets used by this repo. Laptop terms are open-ended, so
# they are built dynamically when needed.
fixed_aspect_maps: dict[str, dict[str, int]] = {
    "SemEval-2014-Restaurant": {
        "food": 0,
        "service": 1,
        "price": 2,
        "ambience": 3,
        "anecdotes/miscellaneous": 4,
    },
    "UIT-VSFC": {
        "lecturer": 0,
        "training_program": 1,
        "facility": 2,
        "others": 3,
    },
}

defaults: dict[str, Any] = {
    "bert_dim": None,
    "max_seq_len": 128,
    "local_context_focus": "cdw",
    "SRD": 3,
    "num_epoch": 5,
    "batch_size": 16,
    "learning_rate": 2.0e-5,
    "l2reg": 1.0e-5,
    "dropout": 0.1,
    "max_grad_norm": 1.0,
    "seed": 42,
    "device": "cuda",
    "use_amp": True,
    "use_gold_aspect_input": True,
    "sentiment_loss_weight": 1.0,
    "aspect_loss_weight": 1.0,
    "monitor_metric": "sentiment_macro_f1",
    "num_workers": 0,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_yaml(path: str | Path) -> SimpleNamespace:
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = {**defaults, **raw}
    required = [
        "dataset",
        "train_file",
        "val_file",
        "pretrained_bert_name",
        "checkpoint_dir",
    ]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    return SimpleNamespace(**cfg)


def _read_aspect_labels(paths: list[str | Path]) -> list[str]:
    labels: set[str] = set()
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    labels.add(rec["gold"]["aspect"])
    return sorted(labels)


def build_aspect_map(cfg: SimpleNamespace) -> dict[str, int]:
    if getattr(cfg, "aspect_map", None):
        return {str(k): int(v) for k, v in cfg.aspect_map.items()}
    if cfg.dataset in fixed_aspect_maps:
        return dict(fixed_aspect_maps[cfg.dataset])
    labels = _read_aspect_labels([cfg.train_file, cfg.val_file])
    if not labels:
        raise ValueError(f"Could not build aspect map from {cfg.train_file} and {cfg.val_file}")
    return {label: idx for idx, label in enumerate(labels)}


def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        first = values[0]
        if isinstance(first, np.ndarray):
            out[key] = torch.tensor(np.stack(values), dtype=torch.long)
        else:
            out[key] = torch.tensor(values, dtype=torch.long)
    return out


def _batch_to_inputs(batch: dict[str, torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    return [
        batch["concat_bert_indices"].to(device),
        batch["concat_segments_indices"].to(device),
        batch["concat_attention_mask"].to(device),
        batch["text_local_indices"].to(device),
        batch["text_local_attention_mask"].to(device),
        batch["aspect_begin"].to(device),
        batch["aspect_len"].to(device),
    ]


@torch.no_grad()
def evaluate_model(model: LCF_BERT, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    sent_gold: list[int] = []
    sent_pred: list[int] = []

    for batch in loader:
        inputs = _batch_to_inputs(batch, device)
        sent_logits, _ = model(inputs)
        sent_gold.extend(batch["polarity"].tolist())
        sent_pred.extend(sent_logits.argmax(dim=-1).cpu().tolist())

    return {
        "sentiment_accuracy": accuracy_score(sent_gold, sent_pred) if sent_gold else 0.0,
        "sentiment_macro_f1": f1_score(sent_gold, sent_pred, average="macro", zero_division=0) if sent_gold else 0.0,
    }


def _resolve_device(name: str) -> torch.device:
    if str(name).lower().startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def _save_checkpoint(
    path: Path,
    model: LCF_BERT,
    epoch: int,
    cfg: SimpleNamespace,
    model_config: dict[str, Any],
    sentiment_map: dict[str, int],
    aspect_map: dict[str, int],
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "model_config": model_config,
            "pretrained_bert_name": cfg.pretrained_bert_name,
            "sentiment_map": sentiment_map,
            "aspect_map": aspect_map,
            "val_metrics": metrics,
            "dataset": cfg.dataset,
        },
        path,
    )


def train(cfg: SimpleNamespace) -> Path:
    set_seed(int(cfg.seed))
    device = _resolve_device(str(cfg.device))
    logger.info("Device: %s", device)

    aspect_map = build_aspect_map(cfg)
    n_sentiments = len(sentiment_map)
    n_aspects = len(aspect_map)
    logger.info("Dataset: %s", cfg.dataset)
    logger.info("Sentiments: %d | Aspects: %d", n_sentiments, n_aspects)
    logger.info("use_gold_aspect_input: %s", bool(cfg.use_gold_aspect_input))

    tokenizer = Tokenizer4Bert(int(cfg.max_seq_len), cfg.pretrained_bert_name)

    train_ds = ABSADatasetJSONL(
        cfg.train_file,
        tokenizer,
        sentiment_map,
        aspect_map,
        use_gold_aspect_input=bool(cfg.use_gold_aspect_input),
    )
    val_ds = ABSADatasetJSONL(
        cfg.val_file,
        tokenizer,
        sentiment_map,
        aspect_map,
        use_gold_aspect_input=bool(cfg.use_gold_aspect_input),
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            f"Empty dataset after filtering. train={len(train_ds)} val={len(val_ds)} "
            f"skipped_train={train_ds.skipped} skipped_val={val_ds.skipped}"
        )
    logger.info(
        "Train: %d | Val: %d | skipped train/val: %d/%d",
        len(train_ds),
        len(val_ds),
        train_ds.skipped,
        val_ds.skipped,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        collate_fn=collate,
        num_workers=int(cfg.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        collate_fn=collate,
        num_workers=int(cfg.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    bert = AutoModel.from_pretrained(cfg.pretrained_bert_name)
    hidden_size = int(getattr(bert.config, "hidden_size", 768))
    if cfg.bert_dim is not None and int(cfg.bert_dim) != hidden_size:
        logger.warning(
            "Config bert_dim=%s differs from model hidden_size=%s. Using hidden_size.",
            cfg.bert_dim,
            hidden_size,
        )

    model_config = {
        "bert_dim": hidden_size,
        "dropout": float(cfg.dropout),
        "max_seq_len": int(cfg.max_seq_len),
        "local_context_focus": str(cfg.local_context_focus).lower(),
        "SRD": int(cfg.SRD),
        "polarities_dim": n_sentiments,
        "aspects_dim": n_aspects,
        "device": str(device),
        "pretrained_bert_name": cfg.pretrained_bert_name,
        "use_gold_aspect_input": bool(cfg.use_gold_aspect_input),
    }
    opt = SimpleNamespace(**model_config)
    model = LCF_BERT(bert, opt).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.l2reg),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(cfg.num_epoch), 1),
    )
    loss_fn = nn.CrossEntropyLoss()
    use_amp = bool(cfg.use_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "lcf_bert_best.pt"
    best_value = -1.0

    sentiment_loss_weight = float(cfg.sentiment_loss_weight)
    aspect_loss_weight = float(cfg.aspect_loss_weight)
    monitor_metric = str(cfg.monitor_metric)

    for epoch in range(1, int(cfg.num_epoch) + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            inputs = _batch_to_inputs(batch, device)
            sent_labels = batch["polarity"].to(device)
            asp_labels = batch["aspect_label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                sent_logits, asp_logits = model(inputs)
                sent_loss = loss_fn(sent_logits, sent_labels)
                asp_loss = loss_fn(asp_logits, asp_labels)
                loss = sentiment_loss_weight * sent_loss + aspect_loss_weight * asp_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.max_grad_norm))
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            if step % 50 == 0 or step == len(train_loader):
                logger.info(
                    "epoch %d/%d step %d/%d loss=%.4f",
                    epoch,
                    int(cfg.num_epoch),
                    step,
                    len(train_loader),
                    total_loss / step,
                )

        scheduler.step()
        metrics = evaluate_model(model, val_loader, device)
        logger.info(
            "Epoch %d/%d | sent_acc=%.4f sent_f1=%.4f",
            epoch,
            int(cfg.num_epoch),
            metrics["sentiment_accuracy"],
            metrics["sentiment_macro_f1"],
        )

        monitor_value = float(metrics.get(monitor_metric, metrics["sentiment_macro_f1"]))
        if monitor_value > best_value:
            best_value = monitor_value
            _save_checkpoint(
                best_path,
                model,
                epoch,
                cfg,
                model_config,
                sentiment_map,
                aspect_map,
                metrics,
            )
            logger.info("Saved best checkpoint: %s (%s=%.4f)", best_path, monitor_metric, best_value)

    logger.info("Training done. Best %s: %.4f", monitor_metric, best_value)
    return best_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train LCF-BERT")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args(argv)
    cfg = _load_yaml(args.config)
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
