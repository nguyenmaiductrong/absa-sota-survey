"""Data utilities for LCF-BERT.

Two dataset classes:
  - ABSADatasetJSONL  : reads the unified JSONL format produced by prepare_*.py
  - ABSADataset       : legacy 3-line format (kept for backward compatibility)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer


# ── Tokeniser helper ─────────────────────────────────────────────────────────

def pad_and_truncate(sequence, maxlen, dtype="int64",
                     padding="post", truncating="post", value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    trunc = sequence[-maxlen:] if truncating == "pre" else sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == "post":
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer4Bert:
    def __init__(self, max_seq_len: int, pretrained_bert_name: str):
        self.tokenizer   = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text: str, reverse: bool = False,
                         padding: str = "post", truncating: str = "post"):
        seq = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(text)
        )
        if not seq:
            seq = [0]
        if reverse:
            seq = seq[::-1]
        return pad_and_truncate(seq, self.max_seq_len,
                                padding=padding, truncating=truncating)


# ── Unified JSONL dataset ─────────────────────────────────────────────────────

class ABSADatasetJSONL(Dataset):
    """Dataset that reads the unified JSONL produced by prepare_*.py.

    Each record must have:
      text, gold.aspect, gold.sentiment
    and optionally a ``label_maps`` dict mapping string labels → int indices.

    Parameters
    ----------
    path : str | Path
        Path to the JSONL file.
    tokenizer : Tokenizer4Bert
    sentiment_map : dict[str, int]
        e.g. {"negative": 0, "neutral": 1, "positive": 2}
    aspect_map : dict[str, int]
        e.g. {"food": 0, "service": 1, ...}
    """

    def __init__(self, path: str | Path, tokenizer: Tokenizer4Bert,
                 sentiment_map: dict[str, int], aspect_map: dict[str, int]):
        self.tokenizer     = tokenizer
        self.sentiment_map = sentiment_map
        self.aspect_map    = aspect_map
        self.data: list[dict] = []
        self._load(Path(path))

    def _load(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text     = rec["text"]
                aspect   = rec["gold"]["aspect"]
                polarity = rec["gold"]["sentiment"]

                if aspect not in self.aspect_map:
                    continue
                if polarity not in self.sentiment_map:
                    continue

                self.data.append(self._encode(text, aspect,
                                              self.sentiment_map[polarity],
                                              self.aspect_map[aspect]))

    def _encode(self, text: str, aspect: str,
                polarity_idx: int, aspect_idx: int) -> dict:
        tok = self.tokenizer
        aspect_indices  = tok.text_to_sequence(aspect)
        aspect_len      = int(np.sum(aspect_indices != 0))

        text_bert_indices = tok.text_to_sequence(
            f"[CLS] {text} [SEP] {aspect} [SEP]"
        )
        text_len = int(np.sum(text_bert_indices != 0))

        concat_bert_indices = tok.text_to_sequence(
            f"[CLS] {text} [SEP] {aspect} [SEP]"
        )
        concat_segments_indices = pad_and_truncate(
            [0] * (text_len + 2) + [1] * (aspect_len + 1),
            tok.max_seq_len,
        )

        text_local_indices = tok.text_to_sequence(f"[CLS] {text} [SEP]")

        return {
            "concat_bert_indices":    concat_bert_indices,
            "concat_segments_indices": concat_segments_indices,
            "text_bert_indices":      text_bert_indices,
            "text_local_indices":     text_local_indices,
            "aspect_indices":         aspect_indices,
            "polarity":               polarity_idx,
            "aspect_label":           aspect_idx,
        }

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


# ── Legacy 3-line dataset (unchanged interface) ───────────────────────────────

class ABSADataset(Dataset):
    """Legacy 3-line format: text_with_$T$, aspect, polarity_int."""

    def __init__(self, fname: str, tokenizer: Tokenizer4Bert):
        with open(fname, "r", encoding="utf-8", newline="\n", errors="ignore") as fh:
            lines = fh.readlines()

        all_data: list[dict] = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [
                s.lower().strip() for s in lines[i].partition("$T$")
            ]
            aspect   = lines[i + 1].lower().strip()
            polarity = int(lines[i + 2].strip()) + 1

            aspect_indices  = tokenizer.text_to_sequence(aspect)
            aspect_len      = int(np.sum(aspect_indices != 0))
            text_bert_indices = tokenizer.text_to_sequence(
                f"[CLS] {text_left} {aspect} {text_right} [SEP]"
            )
            text_len = int(np.sum(text_bert_indices != 0))
            concat_bert_indices = tokenizer.text_to_sequence(
                f"[CLS] {text_left} {aspect} {text_right} [SEP] {aspect} [SEP]"
            )
            concat_segments_indices = pad_and_truncate(
                [0] * (text_len + 2) + [1] * (aspect_len + 1),
                tokenizer.max_seq_len,
            )
            aspect_bert_indices = tokenizer.text_to_sequence(
                f"[CLS] {aspect} [SEP]"
            )
            all_data.append({
                "concat_bert_indices":     concat_bert_indices,
                "concat_segments_indices": concat_segments_indices,
                "text_bert_indices":       text_bert_indices,
                "aspect_bert_indices":     aspect_bert_indices,
                "aspect_indices":          aspect_indices,
                "polarity":                polarity,
                "aspect_label":            0,   # unknown in legacy format
            })

        self.data = all_data

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)
