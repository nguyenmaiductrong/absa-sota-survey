"""
Dataset loader cho InstructABSA — đọc JSONL đã prep ở Phase 1, prepend
instruction prompt (Phase 2), tokenize bằng tokenizer của backbone Seq2SeqLM.

Tách thành module riêng để train.py + predictor reuse được logic format prompt.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .instructions import build_prompt, Language


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Đọc 1 file JSONL → list of dict."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def subsample(records: list[dict[str, Any]],
              ratio: float,
              seed: int = 42) -> list[dict[str, Any]]:
    """Lấy ngẫu nhiên `ratio` (0-1) tỉ lệ records, giữ nguyên thứ tự đầu vào.

    Paper InstructABSA chứng minh 50% data đã đạt kết quả cạnh tranh.
    """
    if ratio >= 1.0:
        return records
    if ratio <= 0.0:
        raise ValueError(f"subsample ratio must be in (0, 1], got {ratio}")
    rng = random.Random(seed)
    k = max(1, int(len(records) * ratio))
    indices = sorted(rng.sample(range(len(records)), k))
    return [records[i] for i in indices]


class InstructABSADataset(Dataset):
    """Dataset wrapper: prompt = build_prompt(input_text, lang); target = output_text."""

    def __init__(self,
                 records: list[dict[str, Any]],
                 tokenizer: PreTrainedTokenizerBase,
                 language: Language,
                 max_input_length: int = 512,
                 max_target_length: int = 8):
        self.records = records
        self.tokenizer = tokenizer
        self.language = language
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        prompt = build_prompt(rec["input_text"], self.language)
        target = rec["output_text"]

        model_inputs = self.tokenizer(
            prompt,
            max_length=self.max_input_length,
            truncation=True,
            padding=False,
        )
        labels = self.tokenizer(
            target,
            max_length=self.max_target_length,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def load_split(jsonl_path: str | Path,
               tokenizer: PreTrainedTokenizerBase,
               language: Language,
               subsample_ratio: float = 1.0,
               max_input_length: int = 512,
               max_target_length: int = 8,
               seed: int = 42) -> InstructABSADataset:
    records = read_jsonl(jsonl_path)
    records = subsample(records, subsample_ratio, seed=seed)
    return InstructABSADataset(
        records=records,
        tokenizer=tokenizer,
        language=language,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
    )
