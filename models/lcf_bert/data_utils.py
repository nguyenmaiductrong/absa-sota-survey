"""Data utilities for the drop-in LCF-BERT implementation.

This file intentionally supports both BERT and PhoBERT/Roberta-style models by
using Hugging Face AutoTokenizer instead of BertTokenizer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def _as_int64_array(values: list[int], max_len: int, pad_value: int = 0) -> np.ndarray:
    arr = np.full(max_len, pad_value, dtype=np.int64)
    values = values[:max_len]
    if values:
        arr[: len(values)] = np.asarray(values, dtype=np.int64)
    return arr


def _pad_token_ids(token_ids: list[int], max_len: int, pad_id: int) -> np.ndarray:
    out = np.full(max_len, pad_id, dtype=np.int64)
    n = min(len(token_ids), max_len)
    if n:
        out[:n] = np.asarray(token_ids[:n], dtype=np.int64)
    return out


def _token_ids_for_text(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))


def _nnz_token_ids(arr: np.ndarray, pad_id: int) -> int:
    return int(np.sum(arr != pad_id))


def semantic_relative_distance(
    aspect_begin: int,
    aspect_len: int,
    text_len: int,
) -> np.ndarray:
    """Per-token SRD (Formula 1): SRD_i = |i - P_a| - floor(m/2).

    P_a is the center position of the aspect span (multi-token aspects);
    m is the aspect length in tokens; i is the token index in the sentence.
    """
    if text_len <= 0:
        return np.zeros(0, dtype=np.float64)
    if aspect_len <= 0:
        return np.full(text_len, np.inf, dtype=np.float64)
    p_a = aspect_begin + (aspect_len - 1) / 2.0
    floor_half_m = aspect_len // 2
    ii = np.arange(text_len, dtype=np.float64)
    return np.abs(ii - p_a) - floor_half_m


def get_lca_ids_and_cdm_vec(
    max_seq_len: int,
    SRD: int,
    bert_spc_indices: np.ndarray,
    aspect_indices: np.ndarray,
    aspect_begin: int,
    pad_id: int,
    syntactical_dist: np.ndarray | None = None,
) -> np.ndarray:
    """CDM: mask 1 if SRD_i <= alpha, else 0 alpha is config SRD."""
    cdm_vec = np.zeros(max_seq_len, dtype=np.int64)
    aspect_len = _nnz_token_ids(aspect_indices, pad_id)
    text_len = _nnz_token_ids(bert_spc_indices, pad_id) - _nnz_token_ids(aspect_indices, pad_id) - 1
    n_use = min(text_len, max_seq_len)
    alpha = float(SRD)
    if syntactical_dist is not None:
        raw = np.asarray(syntactical_dist, dtype=np.float64).ravel()
        dist = np.full(n_use, np.inf, dtype=np.float64)
        dist[: min(n_use, raw.size)] = raw[:n_use]
    else:
        dist = semantic_relative_distance(aspect_begin, aspect_len, text_len)[:n_use]
    for i in range(n_use):
        if dist[i] <= alpha:
            cdm_vec[i] = 1
    return cdm_vec


def get_cdw_vec(
    max_seq_len: int,
    SRD: int,
    bert_spc_indices: np.ndarray,
    aspect_indices: np.ndarray,
    aspect_begin: int,
    pad_id: int,
    syntactical_dist: np.ndarray | None = None,
) -> np.ndarray:
    """CDW: weight 1 if SRD_i <= alpha; else (SRD_i - alpha) / n. n = sentence length."""
    cdw_vec = np.zeros(max_seq_len, dtype=np.float32)
    aspect_len = _nnz_token_ids(aspect_indices, pad_id)
    text_len = _nnz_token_ids(bert_spc_indices, pad_id) - _nnz_token_ids(aspect_indices, pad_id) - 1
    n_use = min(text_len, max_seq_len)
    alpha = float(SRD)
    n = float(text_len) if text_len > 0 else 1.0
    if syntactical_dist is not None:
        raw = np.asarray(syntactical_dist, dtype=np.float64).ravel()
        dist = np.full(n_use, np.inf, dtype=np.float64)
        dist[: min(n_use, raw.size)] = raw[:n_use]
    else:
        dist = semantic_relative_distance(aspect_begin, aspect_len, text_len)[:n_use]
    for i in range(n_use):
        s = float(dist[i])
        if s <= alpha:
            cdw_vec[i] = np.float32(1.0)
        else:
            cdw_vec[i] = np.float32((s - alpha) / n)
    return cdw_vec


def _find_subsequence(source: list[int], target: list[int], valid_len: int) -> int:
    """Return start index of target in source[:valid_len], or -1 if missing."""
    if not target:
        return -1
    haystack = source[:valid_len]
    n = len(target)
    for start in range(0, len(haystack) - n + 1):
        if haystack[start : start + n] == target:
            return start
    return -1


class Tokenizer4Bert:
    """Tokenizer wrapper that works for BERT, RoBERTa, and PhoBERT.

    The original project used BertTokenizer and manually inserted [CLS]/[SEP].
    That breaks for PhoBERT because it is Roberta-based and uses different
    special tokens. This wrapper lets the tokenizer add model-specific special
    tokens and returns all tensors needed by LCF-BERT.
    """

    def __init__(
        self,
        max_seq_len: int,
        pretrained_bert_name: str,
        srd: int = 3,
        local_context_focus: str = "cdw",
    ):
        self.max_seq_len = int(max_seq_len)
        self.pretrained_bert_name = pretrained_bert_name
        self.srd = int(srd)
        self.local_context_focus = str(local_context_focus).lower()
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            pretrained_bert_name,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.sep_token
        self.pad_token_id = int(self.tokenizer.pad_token_id or 0)

    def _encode_single(self, text: str) -> dict[str, np.ndarray]:
        enc = self.tokenizer(
            str(text),
            add_special_tokens=True,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        input_ids = _as_int64_array(enc["input_ids"], self.max_seq_len, self.pad_token_id)
        attention_mask = _as_int64_array(enc["attention_mask"], self.max_seq_len, 0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def _encode_pair(self, text: str, aspect: str) -> dict[str, np.ndarray]:
        enc = self.tokenizer(
            str(text),
            str(aspect),
            add_special_tokens=True,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        input_ids = _as_int64_array(enc["input_ids"], self.max_seq_len, self.pad_token_id)
        attention_mask = _as_int64_array(enc["attention_mask"], self.max_seq_len, 0)
        token_type_ids = _as_int64_array(
            enc.get("token_type_ids", [0] * self.max_seq_len),
            self.max_seq_len,
            0,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def encode_for_lcf(
        self,
        text: str,
        aspect: str | None,
        *,
        use_aspect_input: bool,
    ) -> dict[str, np.ndarray | np.int64]:
        """Encode a sample for LCF-BERT.

        Parameters
        ----------
        text:
            Sentence/review text.
        aspect:
            Gold target / category (có thể chuỗi rỗng khi không ghép aspect).
        use_aspect_input:
            If True and aspect is non-empty, encode text-aspect pair. If False,
            encode text only, suitable for text-only joint aspect+sentiment runs.
        """
        text = str(text)
        aspect = "" if aspect is None else str(aspect)
        has_aspect = bool(aspect.strip()) and bool(use_aspect_input)

        if has_aspect:
            pair = self._encode_pair(text, aspect)
        else:
            single = self._encode_single(text)
            pair = {
                "input_ids": single["input_ids"],
                "attention_mask": single["attention_mask"],
                "token_type_ids": np.zeros(self.max_seq_len, dtype=np.int64),
            }

        local = self._encode_single(text)

        pad_id = self.pad_token_id
        if not has_aspect:
            lcf_w = np.zeros(self.max_seq_len, dtype=np.float32)
            vlen = int(local["attention_mask"].sum())
            lcf_w[:vlen] = 1.0
            aspect_plain_ids: list[int] = []
            aspect_begin = np.int64(-1)
            aspect_len = np.int64(0)
        else:
            asp = aspect.strip()
            pos = text.find(asp)
            if pos >= 0:
                text_left = text[:pos]
                text_right = text[pos + len(asp) :]
            else:
                text_left = ""
                text_right = text
            parts = [text_left.strip(), asp, text_right.strip()]
            text_raw = " ".join(p for p in parts if p).strip()
            if not text_raw:
                text_raw = text.strip()

            bos = self.tokenizer.bos_token or "[CLS]"
            eos = self.tokenizer.sep_token or "[SEP]"
            text_spc_str = f"{bos} {text_raw} {eos} {asp} {eos}"
            spc_ids = _token_ids_for_text(self.tokenizer, text_spc_str)
            text_indices = _pad_token_ids(spc_ids, self.max_seq_len, pad_id)

            aspect_sub_ids = self.tokenizer.encode(asp, add_special_tokens=False)
            aspect_indices = _pad_token_ids(aspect_sub_ids, self.max_seq_len, pad_id)

            aspect_begin_py = len(self.tokenizer.tokenize(bos + " " + text_left))

            if pos < 0:
                local_ids_list = local["input_ids"].astype(int).tolist()
                local_valid_len = int(local["attention_mask"].sum())
                fb = _find_subsequence(local_ids_list, aspect_sub_ids, local_valid_len)
                if fb >= 0:
                    aspect_begin_py = fb

            if self.local_context_focus == "cdm":
                raw_vec = get_lca_ids_and_cdm_vec(
                    self.max_seq_len,
                    self.srd,
                    text_indices,
                    aspect_indices,
                    aspect_begin_py,
                    pad_id,
                    None,
                )
                lcf_w = raw_vec.astype(np.float32)
            else:
                lcf_w = get_cdw_vec(
                    self.max_seq_len,
                    self.srd,
                    text_indices,
                    aspect_indices,
                    aspect_begin_py,
                    pad_id,
                    None,
                )

            aspect_plain_ids = aspect_sub_ids
            local_ids_list = local["input_ids"].astype(int).tolist()
            local_valid_len = int(local["attention_mask"].sum())
            aspect_begin = np.int64(
                _find_subsequence(local_ids_list, aspect_plain_ids, local_valid_len)
            )
            aspect_len = np.int64(len(aspect_plain_ids) if aspect_begin >= 0 else 0)

        return {
            "concat_bert_indices": pair["input_ids"],
            "concat_segments_indices": pair["token_type_ids"],
            "concat_attention_mask": pair["attention_mask"],
            "text_local_indices": local["input_ids"],
            "text_local_attention_mask": local["attention_mask"],
            "lcf_context_weight": lcf_w,
            "aspect_begin": aspect_begin,
            "aspect_len": aspect_len,
        }


class ABSADatasetJSONL(Dataset):
    """Dataset for the unified JSONL files in data/processed/lcf_bert.

    Each JSONL record must contain:
      - text
      - gold.aspect
      - gold.sentiment
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: Tokenizer4Bert,
        sentiment_map: dict[str, int],
        aspect_map: dict[str, int],
        use_gold_aspect_input: bool,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.sentiment_map = sentiment_map
        self.aspect_map = aspect_map
        self.use_gold_aspect_input = bool(use_gold_aspect_input)
        self.data: list[dict[str, Any]] = []
        self.skipped = 0
        self._load()

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = rec["text"]
                aspect = rec["gold"]["aspect"]
                sentiment = rec["gold"]["sentiment"]

                if aspect not in self.aspect_map or sentiment not in self.sentiment_map:
                    self.skipped += 1
                    continue

                features = self.tokenizer.encode_for_lcf(
                    text,
                    aspect=aspect,
                    use_aspect_input=self.use_gold_aspect_input,
                )
                features["polarity"] = np.int64(self.sentiment_map[sentiment])
                features["aspect_label"] = np.int64(self.aspect_map[aspect])
                self.data.append(features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


