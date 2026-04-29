"""DOT Predictor stub — implements the Predictor protocol.

Paper: "DOT: An efficient Double Transformer for NLP tasks with tables"
       (used here in the ABSA context as a generative seq2seq approach)
Paradigm: Generative-Seq2Seq

Replace the body of predict() with actual model inference.

Usage (via evaluate.py CLI):
  python evaluate.py \
      --predictor predictors.dot:DOTPredictor \
      --predictor-kwargs '{"checkpoint": "checkpoints/dot/best.pt"}' \
      --test-set data/processed/lcf_bert/semeval14_rest_test.jsonl
"""
from __future__ import annotations

PARSE_ERROR = "__PARSE_ERROR__"


class DOTPredictor:
    method   = "DOT"
    paradigm = "Generative-Seq2Seq"
    backbone = "t5-base"

    def __init__(self, checkpoint: str | None = None, device: str = "cuda"):
        self._device = device
        self._checkpoint = checkpoint

    def warmup(self, text: str) -> None:
        self.predict(text)

    def predict(self, text: str, aspect: str | None = None) -> tuple[str, str, str]:
        return PARSE_ERROR, PARSE_ERROR, "<<DOT not implemented>>"
