"""LCF-BERT Predictor — implements the Predictor protocol defined in evaluate.py.

Load a trained checkpoint and expose predict(text) -> (aspect, sentiment, raw).

Usage (via evaluate.py CLI):
  python evaluate.py \
      --predictor predictors.lcf_bert:LCFBertPredictor \
      --predictor-kwargs '{"checkpoint": "checkpoints/semeval14_rest/lcf_bert_best.pt"}' \
      --test-set data/processed/lcf_bert/semeval14_rest_test.jsonl \
      --output-dir results
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import BertModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.lcf_bert.data_utils import Tokenizer4Bert, pad_and_truncate
from models.lcf_bert.lcf_bert import LCF_BERT

import numpy as np

PARSE_ERROR = "__PARSE_ERROR__"


class LCFBertPredictor:
    """Wraps a trained LCF-BERT checkpoint for evaluation.

    Parameters
    ----------
    checkpoint : str
        Path to the .pt file saved by models/lcf_bert/train.py
    device : str, optional
        "cuda" or "cpu" (auto-detected when omitted)
    """

    method   = "LCF-BERT"
    paradigm = "Discriminative"
    backbone = "bert-base-uncased"   # overridden from checkpoint if available

    def __init__(self, checkpoint: str, device: str | None = None):
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        self._device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._load(ckpt_path)

    def _load(self, ckpt_path: Path) -> None:
        ckpt = torch.load(ckpt_path, map_location=self._device)

        self._sentiment_map: dict[str, int] = ckpt["sentiment_map"]
        self._aspect_map:    dict[str, int] = ckpt["aspect_map"]

        # reverse maps: int → label string
        self._idx2sentiment = {v: k for k, v in self._sentiment_map.items()}
        self._idx2aspect    = {v: k for k, v in self._aspect_map.items()}

        raw_opt = ckpt["opt"]
        opt = SimpleNamespace(**raw_opt)

        # Update class attributes from checkpoint
        pretrained = raw_opt.get("pretrained_bert_name",
                                 getattr(opt, "pretrained_bert_name", "bert-base-uncased"))
        self.__class__.backbone = pretrained

        self._tokenizer = Tokenizer4Bert(opt.max_seq_len, pretrained)
        self._opt = opt

        bert = BertModel.from_pretrained(pretrained)
        self._model = LCF_BERT(bert, opt).to(self._device)
        self._model.load_state_dict(ckpt["model_state"])
        self._model.eval()

    # ── Predictor protocol ────────────────────────────────────────────────────

    def warmup(self, text: str) -> None:
        """JIT warm-up: run predict without timing."""
        self.predict(text)

    @torch.no_grad()
    def predict(self, text: str, aspect: str | None = None) -> tuple[str, str, str]:
        """Predict (aspect_category, sentiment, raw_json) from text alone."""
        try:
            inputs = self._encode(text)
            sent_logits, asp_logits = self._model(inputs)

            sent_idx = int(sent_logits.argmax(dim=-1).item())
            asp_idx  = int(asp_logits.argmax(dim=-1).item())

            pred_sentiment = self._idx2sentiment.get(sent_idx, PARSE_ERROR)
            pred_aspect    = self._idx2aspect.get(asp_idx,    PARSE_ERROR)

            sent_probs = torch.softmax(sent_logits, dim=-1).squeeze().tolist()
            asp_probs  = torch.softmax(asp_logits,  dim=-1).squeeze().tolist()

            raw = json.dumps({
                "pred_sentiment": pred_sentiment,
                "pred_aspect":    pred_aspect,
                "sent_probs":     sent_probs,
                "asp_probs":      asp_probs,
            }, ensure_ascii=False)

            return pred_aspect, pred_sentiment, raw

        except Exception as exc:  # noqa: BLE001
            return PARSE_ERROR, PARSE_ERROR, f"<<error: {exc}>>"

    # ── Encoding helper ────────────────────────────────────────────────────────

    def _encode(self, text: str) -> list[torch.Tensor]:
        """Encode text with a dummy aspect token (empty string for inference)."""
        tok = self._tokenizer
        dummy_aspect = ""   # model predicts aspect — no aspect given at inference

        aspect_indices = tok.text_to_sequence(dummy_aspect if dummy_aspect else "[UNK]")
        aspect_len     = max(1, int(np.sum(aspect_indices != 0)))

        text_bert_indices = tok.text_to_sequence(f"[CLS] {text} [SEP]")
        text_len = int(np.sum(text_bert_indices != 0))

        concat_bert_indices = tok.text_to_sequence(
            f"[CLS] {text} [SEP]"
        )
        concat_segments_indices = pad_and_truncate(
            [0] * tok.max_seq_len, tok.max_seq_len
        )
        text_local_indices = tok.text_to_sequence(f"[CLS] {text} [SEP]")

        def _t(arr) -> torch.Tensor:
            return torch.tensor(arr, dtype=torch.long).unsqueeze(0).to(self._device)

        return [
            _t(concat_bert_indices),
            _t(concat_segments_indices),
            _t(text_local_indices),
            _t(aspect_indices),
        ]
