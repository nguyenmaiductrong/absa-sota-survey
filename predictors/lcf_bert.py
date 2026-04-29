"""LCF-BERT Predictor for evaluate.py.

Usage:
  python evaluate.py \
    --predictor predictors.lcf_bert:LCFBertPredictor \
    --predictor-kwargs '{"checkpoint":"checkpoints/semeval14_rest/lcf_bert_best.pt"}' \
    --test-set data/processed/lcf_bert/semeval14_rest_test.jsonl \
    --given-aspect \
    --output-dir results
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.schema import PARSE_ERROR_TOKEN as PARSE_ERROR
from models.lcf_bert.data_utils import Tokenizer4Bert
from models.lcf_bert.lcf_bert import LCF_BERT


def _torch_load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load both new and older project checkpoints."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class LCFBertPredictor:
    method = "LCF-BERT"
    paradigm = "Discriminative"
    backbone = "unknown"

    def __init__(self, checkpoint: str, device: str | None = None):
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        self._device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._load(ckpt_path)

    def _load(self, ckpt_path: Path) -> None:
        ckpt = _torch_load_checkpoint(ckpt_path, self._device)
        self._sentiment_map: dict[str, int] = ckpt["sentiment_map"]
        self._aspect_map: dict[str, int] = ckpt["aspect_map"]
        self._idx2sentiment = {int(v): str(k) for k, v in self._sentiment_map.items()}
        self._idx2aspect = {int(v): str(k) for k, v in self._aspect_map.items()}

        raw_opt = dict(ckpt.get("model_config") or {})
        pretrained = (
            raw_opt.get("pretrained_bert_name")
            or ckpt.get("pretrained_bert_name")
            or "bert-base-uncased"
        )
        raw_opt["pretrained_bert_name"] = pretrained
        raw_opt["device"] = str(self._device)
        raw_opt.setdefault("use_gold_aspect_input", False)
        raw_opt.setdefault("dropout", 0.1)
        raw_opt.setdefault("local_context_focus", "cdw")
        raw_opt.setdefault("SRD", 3)
        raw_opt.setdefault("polarities_dim", len(self._sentiment_map))
        raw_opt.setdefault("aspects_dim", len(self._aspect_map))
        raw_opt.setdefault("max_seq_len", 128)

        mu_log = logging.getLogger("transformers.modeling_utils")
        _lvl = mu_log.level
        mu_log.setLevel(logging.ERROR)
        try:
            bert = AutoModel.from_pretrained(pretrained)
        finally:
            mu_log.setLevel(_lvl)
        raw_opt["bert_dim"] = int(raw_opt.get("bert_dim") or getattr(bert.config, "hidden_size", 768))
        opt = SimpleNamespace(**raw_opt)

        self.backbone = str(pretrained)
        self._use_gold_aspect_input = bool(raw_opt.get("use_gold_aspect_input", False))
        self._tokenizer = Tokenizer4Bert(int(opt.max_seq_len), str(pretrained))
        self._model = LCF_BERT(bert, opt).to(self._device)
        self._model.load_state_dict(ckpt["model_state"], strict=True)
        self._model.eval()

    def warmup(self, text: str, aspect: str | None = None) -> None:
        self.predict(text, aspect=aspect)

    @torch.no_grad()
    def predict(self, text: str, aspect: str | None = None) -> tuple[str, str, str]:
        """Return (pred_aspect, pred_sentiment, raw_json).

        If aspect is supplied by evaluate.py --given-aspect, the model uses it
        as the LCF target and returns that same aspect label. If aspect is not
        supplied, the aspect head is used.
        """
        try:
            given_aspect = aspect.strip() if isinstance(aspect, str) and aspect.strip() else ""
            use_aspect_input = bool(given_aspect)
            features = self._tokenizer.encode_for_lcf(
                text,
                aspect=given_aspect,
                use_aspect_input=use_aspect_input,
            )
            inputs = [
                self._tensor(features["concat_bert_indices"]),
                self._tensor(features["concat_segments_indices"]),
                self._tensor(features["concat_attention_mask"]),
                self._tensor(features["text_local_indices"]),
                self._tensor(features["text_local_attention_mask"]),
                self._tensor_scalar(features["aspect_begin"]),
                self._tensor_scalar(features["aspect_len"]),
            ]

            sent_logits, asp_logits = self._model(inputs)
            sent_idx = int(sent_logits.argmax(dim=-1).item())
            asp_idx = int(asp_logits.argmax(dim=-1).item())

            pred_sentiment = self._idx2sentiment.get(sent_idx, PARSE_ERROR)
            pred_aspect = given_aspect if given_aspect else self._idx2aspect.get(asp_idx, PARSE_ERROR)

            raw = json.dumps(
                {
                    "given_aspect": given_aspect or None,
                    "pred_aspect": pred_aspect,
                    "pred_sentiment": pred_sentiment,
                    "sent_probs": torch.softmax(sent_logits, dim=-1).squeeze(0).detach().cpu().tolist(),
                    "asp_probs": torch.softmax(asp_logits, dim=-1).squeeze(0).detach().cpu().tolist(),
                },
                ensure_ascii=False,
            )
            return pred_aspect, pred_sentiment, raw
        except Exception as exc:  # noqa: BLE001
            return PARSE_ERROR, PARSE_ERROR, f"<<error: {type(exc).__name__}: {exc}>>"

    def _tensor(self, arr: np.ndarray | Any) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.long).unsqueeze(0).to(self._device)

    def _tensor_scalar(self, value: np.integer | int | Any) -> torch.Tensor:
        return torch.tensor([int(value)], dtype=torch.long).to(self._device)
