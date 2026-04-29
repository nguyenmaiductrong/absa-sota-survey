"""SSIN Predictor stub — implements the Predictor protocol.

Paper: "Span-level Sentiment Interaction Network for Aspect-Based Sentiment Analysis"
Paradigm: Graph (span-level interaction)

Replace the body of predict() with actual model inference.

Usage (via evaluate.py CLI):
  python evaluate.py \
      --predictor predictors.ssin:SSINPredictor \
      --predictor-kwargs '{"checkpoint": "checkpoints/ssin/best.pt"}' \
      --test-set data/processed/lcf_bert/semeval14_rest_test.jsonl
"""
from __future__ import annotations

PARSE_ERROR = "__PARSE_ERROR__"


class SSINPredictor:
    method   = "SSIN"
    paradigm = "Graph"
    backbone = "bert-base-uncased"

    def __init__(self, checkpoint: str | None = None, device: str = "cuda"):
        self._device = device
        self._checkpoint = checkpoint

    def warmup(self, text: str) -> None:
        self.predict(text)

    def predict(self, text: str, aspect: str | None = None) -> tuple[str, str, str]:
        return PARSE_ERROR, PARSE_ERROR, "<<SSIN not implemented>>"
