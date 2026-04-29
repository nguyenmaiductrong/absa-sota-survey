"""InstructABSA Predictor stub — implements the Predictor protocol.

Paper: "InstructABSA: Instruction Learning for Aspect Based Sentiment Analysis"
Paradigm: Instruction-Tuning (T5 / Flan-T5)

Replace the body of predict() with actual model inference once the
InstructABSA checkpoint is available.

Usage (via evaluate.py CLI):
  python evaluate.py \
      --predictor predictors.instruct_absa:InstructABSAPredictor \
      --predictor-kwargs '{"model_name_or_path": "kevinscaria/instruct_absa_tk-instruct-base-def-pos-neg-neut-combined"}' \
      --test-set data/processed/lcf_bert/semeval14_rest_test.jsonl
"""
from __future__ import annotations

PARSE_ERROR = "__PARSE_ERROR__"


class InstructABSAPredictor:
    method   = "InstructABSA"
    paradigm = "Instruction-Tuning"
    backbone = "kevinscaria/instruct_absa_tk-instruct-base-def-pos-neg-neut-combined"

    def __init__(self, model_name_or_path: str | None = None, device: str = "cuda"):
        if model_name_or_path:
            self.backbone = model_name_or_path
        self._device = device

    def warmup(self, text: str) -> None:
        self.predict(text)

    def predict(self, text: str, aspect: str | None = None) -> tuple[str, str, str]:
        return PARSE_ERROR, PARSE_ERROR, "<<InstructABSA not implemented>>"
