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
        # TODO: load model and tokenizer here
        # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        # self._tokenizer = AutoTokenizer.from_pretrained(self.backbone)
        # self._model = AutoModelForSeq2SeqLM.from_pretrained(self.backbone).to(device)

    def warmup(self, text: str) -> None:
        self.predict(text)

    def predict(self, text: str, aspect: str | None = None) -> tuple[str, str, str]:
        # TODO: replace stub with real inference
        # prompt = f"What is the aspect and sentiment of: '{text}'"
        # inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        # output = self._model.generate(**inputs, max_new_tokens=64)
        # decoded = self._tokenizer.decode(output[0], skip_special_tokens=True)
        # pred_aspect, pred_sentiment = _parse_output(decoded)
        # return pred_aspect, pred_sentiment, decoded
        return PARSE_ERROR, PARSE_ERROR, "<<InstructABSA not implemented>>"
