from __future__ import annotations
import os, sys, torch

PARSE_ERROR = "__PARSE_ERROR__"

SENT_MAP = {
    "great": "positive",
    "ok": "neutral",
    "bad": "negative",
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
}

class DOTPredictor:
    method   = "DOT"
    paradigm = "Generative-Seq2Seq"
    backbone = "t5-base"

    def __init__(
        self,
        checkpoint: str | None = None,
        checkpoint_phase1: str = "/kaggle/working/checkpoints/dot_rest/first",
        checkpoint_phase2: str = "/kaggle/working/checkpoints/dot_rest/final",
        device: str = "cuda",
    ):
        model_dir = "/kaggle/working/absa-sota-survey/models/dot"
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        from transformers import T5Tokenizer
        from t5 import MyT5ForConditionalGeneration
        from eval_utils import extract_spans_para

        self._extract_spans_para = extract_spans_para
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.tokenizer = T5Tokenizer.from_pretrained(checkpoint_phase2, local_files_only=True)

        self._model1 = MyT5ForConditionalGeneration.from_pretrained(checkpoint_phase1, local_files_only=True)
        self._model1 = self._model1.to(self._device)
        self._model1.eval()

        self._model2 = MyT5ForConditionalGeneration.from_pretrained(checkpoint_phase2, local_files_only=True)
        self._model2 = self._model2.to(self._device)
        self._model2.eval()

        print(f"✅ DOTPredictor loaded | device={self._device}")

    def _generate(self, model, text: str, max_length: int = 200) -> str:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=200,
        ).to(self._device)

        with torch.no_grad():
            out = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_length=max_length,
                num_beams=1,
                early_stopping=True,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def warmup(self, text: str = "The food was great.") -> None:
        self.predict(text)

    def predict(self, text: str, aspect: str | None = None) -> tuple[str, str, str]:
        try:
            # Phase 1: sinh order string
            order_str = self._generate(self._model1, text, max_length=130)

            # Phase 2: sinh full quads
            phase2_input = f"{text} {order_str}"
            raw_output = self._generate(self._model2, phase2_input, max_length=1024)

            quads = self._extract_spans_para(seq=raw_output, seq_type="pred")
            if not quads:
                return PARSE_ERROR, PARSE_ERROR, raw_output

            # Nếu có given aspect, tìm quad khớp
            if aspect:
                for ac, at, sp, ot in quads:
                    if aspect.lower() in (at or "").lower() or aspect.lower() in (ac or "").lower():
                        pred_aspect    = at if at and at != "it" else ac
                        pred_sentiment = SENT_MAP.get(sp.lower().strip() if sp else "", "neutral")
                        return pred_aspect, pred_sentiment, raw_output

            ac, at, sp, ot = quads[0]
            pred_aspect    = at if at and at != "it" else ac
            pred_sentiment = SENT_MAP.get(sp.lower().strip() if sp else "", "neutral")

            return pred_aspect, pred_sentiment, raw_output

        except Exception as e:
            return PARSE_ERROR, PARSE_ERROR, str(e)
