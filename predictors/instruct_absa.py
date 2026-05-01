"""InstructABSA Predictor — implements the Predictor protocol (ATSC subtask).

Paper: "InstructABSA: Instruction Learning for Aspect Based Sentiment Analysis"
       (Scaria et al., NAACL 2024)
Paradigm: Instruction-Tuning (Seq2SeqLM)

Hỗ trợ 2 backbone:
  - EN (SemEval-14): allenai/tk-instruct-base-def-pos
  - VI (UIT-VSFC):   VietAI/vit5-base

Chế độ vận hành: ATSC (given-aspect) — gold aspect được cung cấp ở evaluate.py
qua flag --given-aspect; predictor echo aspect, chỉ predict polarity.

Usage (via evaluate.py CLI):
  python evaluate.py \
      --predictor predictors.instruct_absa:InstructABSAPredictor \
      --predictor-kwargs '{"checkpoint": "checkpoints/semeval14_rest/instruct_absa_best", "language": "en"}' \
      --test-set data/processed/lcf_bert/semeval14_rest_test.jsonl \
      --given-aspect --output-dir results
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Cho phép import package `models` khi predictor được load qua importlib
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.schema import PARSE_ERROR_TOKEN as PARSE_ERROR
from models.instruct_absa.instructions import build_prompt

VALID_POLARITIES = ("positive", "negative", "neutral")
POLARITY_ALIASES = {
    "pos": "positive", "positive": "positive", "tích cực": "positive",
    "neg": "negative", "negative": "negative", "tiêu cực": "negative",
    "neu": "neutral",  "neutral": "neutral",  "trung tính": "neutral",
}


def _resolve_device(name: str | None) -> torch.device:
    if name is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = name.lower()
    if n.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _normalize_polarity(text: str) -> str:
    """Map raw model output → canonical polarity, trả PARSE_ERROR nếu không nhận diện."""
    cleaned = text.strip().lower().rstrip(".!?,; ")
    if cleaned in VALID_POLARITIES:
        return cleaned
    if cleaned in POLARITY_ALIASES:
        return POLARITY_ALIASES[cleaned]
    # Fallback: tìm token polarity đầu tiên xuất hiện
    for token in cleaned.replace(",", " ").split():
        token = token.strip(".!?;:")
        if token in VALID_POLARITIES:
            return token
        if token in POLARITY_ALIASES:
            return POLARITY_ALIASES[token]
    return PARSE_ERROR


class InstructABSAPredictor:
    method   = "InstructABSA"
    paradigm = "Instruction-Tuning"
    backbone = "unknown"

    def __init__(self,
                 checkpoint: str,
                 language: str = "en",
                 device: str | None = None,
                 max_input_length: int = 512,
                 max_new_tokens: int = 8,
                 num_beams: int = 1):
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        if language not in ("en", "vi"):
            raise ValueError(f"language must be 'en' or 'vi', got {language!r}")

        self._language = language
        self._max_input_length = int(max_input_length)
        self._max_new_tokens = int(max_new_tokens)
        self._num_beams = int(num_beams)
        self._device = _resolve_device(device)

        self._tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
        self._model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt_path)).to(self._device)
        self._model.eval()

        # Cập nhật backbone dựa trên config thực tế của model đã load
        try:
            base = self._model.config.name_or_path or str(ckpt_path)
            self.backbone = base
        except AttributeError:
            self.backbone = str(ckpt_path)

    def warmup(self, text: str, aspect: str | None = None) -> None:
        self.predict(text, aspect=aspect)

    @torch.no_grad()
    def predict(self, text: str, aspect: str | None = None) -> tuple[str, str, str]:
        """Return (pred_aspect, pred_sentiment, raw_output).

        ATSC mode: aspect bắt buộc — predictor echo aspect, chỉ generate polarity.
        """
        try:
            given_aspect = aspect.strip() if isinstance(aspect, str) and aspect.strip() else ""
            if not given_aspect:
                return PARSE_ERROR, PARSE_ERROR, "<<missing aspect input for ATSC>>"

            input_text = f"{text.strip()} | {given_aspect}"
            prompt = build_prompt(input_text, self._language)  # type: ignore[arg-type]

            enc = self._tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self._max_input_length,
                truncation=True,
            ).to(self._device)

            gen = self._model.generate(
                **enc,
                max_new_tokens=self._max_new_tokens,
                num_beams=self._num_beams,
                do_sample=False,
            )
            raw_output = self._tokenizer.decode(gen[0], skip_special_tokens=True)
            polarity = _normalize_polarity(raw_output)

            if polarity == PARSE_ERROR:
                return PARSE_ERROR, PARSE_ERROR, raw_output

            raw_json = json.dumps({
                "given_aspect": given_aspect,
                "pred_aspect": given_aspect,
                "pred_sentiment": polarity,
                "raw_decode": raw_output,
            }, ensure_ascii=False)
            return given_aspect, polarity, raw_json

        except Exception as exc:  # noqa: BLE001
            return PARSE_ERROR, PARSE_ERROR, f"<<error: {type(exc).__name__}: {exc}>>"
