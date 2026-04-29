"""LLM-Reasoning Predictor — implements the Predictor protocol.

Paradigm: LLM-Reasoning (zero-shot / few-shot prompting with a large LLM)

Two backends are supported via the ``backend`` constructor arg:
  - "openai"  : OpenAI ChatCompletion API (gpt-4o, gpt-3.5-turbo, …)
  - "ollama"  : local Ollama server       (llama3, mistral, …)

The model answers in a structured JSON format:
  {"aspect": "<label>", "sentiment": "positive|negative|neutral"}

Usage (via evaluate.py CLI):
  # OpenAI
  python evaluate.py \
      --predictor predictors.llm_reasoning:LLMReasoningPredictor \
      --predictor-kwargs '{"backend":"openai","model":"gpt-4o","api_key":"sk-..."}' \
      --test-set data/processed/lcf_bert/semeval14_rest_test.jsonl

  # Ollama (local)
  python evaluate.py \
      --predictor predictors.llm_reasoning:LLMReasoningPredictor \
      --predictor-kwargs '{"backend":"ollama","model":"llama3"}' \
      --test-set data/processed/lcf_bert/vsfc_test.jsonl
"""
from __future__ import annotations

import json
import re

PARSE_ERROR = "__PARSE_ERROR__"

_SYSTEM_PROMPT = (
    "You are an expert in Aspect-Based Sentiment Analysis (ABSA). "
    "Given a sentence, identify:\n"
    "  1. The main ASPECT or TOPIC (e.g. food, service, price, lecturer, facility).\n"
    "  2. The SENTIMENT toward that aspect: positive, negative, or neutral.\n\n"
    "Reply with ONLY a JSON object on a single line:\n"
    '{"aspect": "<aspect>", "sentiment": "<positive|negative|neutral>"}'
)


def _parse_llm_output(raw: str) -> tuple[str, str]:
    """Extract (aspect, sentiment) from a JSON-formatted LLM response."""
    try:
        obj = json.loads(raw.strip())
        return str(obj.get("aspect", PARSE_ERROR)), str(obj.get("sentiment", PARSE_ERROR))
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{[^}]+\}', raw)
    if m:
        try:
            obj = json.loads(m.group())
            return str(obj.get("aspect", PARSE_ERROR)), str(obj.get("sentiment", PARSE_ERROR))
        except json.JSONDecodeError:
            pass
    return PARSE_ERROR, PARSE_ERROR


class LLMReasoningPredictor:
    method   = "LLM-Reasoning"
    paradigm = "LLM-Reasoning"
    backbone = "gpt-4o"

    def __init__(
        self,
        backend: str = "openai",
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 64,
    ):
        self.backbone    = model
        self._backend    = backend
        self._model      = model
        self._api_key    = api_key
        self._base_url   = base_url
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client     = None

        if backend == "openai":
            self._init_openai()
        elif backend == "ollama":
            pass
        else:
            raise ValueError(f"Unknown backend {backend!r}. Use 'openai' or 'ollama'.")

    def _init_openai(self) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ImportError("pip install openai") from exc
        self._client = openai.OpenAI(api_key=self._api_key)

    def warmup(self, text: str) -> None:
        self.predict(text)

    def predict(self, text: str, aspect: str | None = None) -> tuple[str, str, str]:
        try:
            raw = self._call_llm(text)
            pred_aspect, pred_sentiment = _parse_llm_output(raw)
            if pred_aspect == PARSE_ERROR or pred_sentiment == PARSE_ERROR:
                return PARSE_ERROR, PARSE_ERROR, raw
            return pred_aspect, pred_sentiment, raw
        except Exception as exc:  # noqa: BLE001
            return PARSE_ERROR, PARSE_ERROR, f"<<error: {exc}>>"

    def _call_llm(self, text: str) -> str:
        if self._backend == "openai":
            return self._call_openai(text)
        return self._call_ollama(text)

    def _call_openai(self, text: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _call_ollama(self, text: str) -> str:
        import urllib.request
        payload = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
            "stream": False,
            "options": {"temperature": self._temperature},
        }).encode()
        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode())
        return result["message"]["content"].strip()
