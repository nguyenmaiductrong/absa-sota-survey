from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """Light self-attention block used after local/global feature fusion."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = torch.bmm(q, k.transpose(1, 2)) / (self.hidden_size ** 0.5)
        if attention_mask is not None:
            key_mask = attention_mask.unsqueeze(1).bool()
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.bmm(attn, v)

        if attention_mask is not None:
            out = out * attention_mask.unsqueeze(-1).to(out.dtype)
        return self.tanh(out)


class LCF_BERT(nn.Module):
    """LCF-BERT with two heads: sentiment and aspect category.

    Inputs (8 tensors):
      0–4 as before; 5 aspect_begin; 6 aspect_len (aux); 7 lcf_context_weight [B,L] float
      (PyABSA CDM/CDW precomputed in the dataset).
    """

    def __init__(self, bert: nn.Module, opt: SimpleNamespace):
        super().__init__()
        self.bert_spc = bert
        self.bert_local = bert
        self.opt = opt

        hidden_size = int(opt.bert_dim)
        self.dropout = nn.Dropout(float(opt.dropout))
        self.bert_SA = SelfAttention(hidden_size)
        self.linear_cat = nn.Linear(hidden_size * 2, hidden_size)

        self.sentiment_head = nn.Linear(hidden_size, int(opt.polarities_dim))
        self.aspect_head = nn.Linear(hidden_size, int(opt.aspects_dim))

        try:
            self._accepts_token_type_ids = "token_type_ids" in inspect.signature(bert.forward).parameters
        except (TypeError, ValueError):
            self._accepts_token_type_ids = True

    def _bert_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None and self._accepts_token_type_ids:
            kwargs["token_type_ids"] = token_type_ids
        return self.bert_spc(**kwargs).last_hidden_state

    @staticmethod
    def _masked_mean(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        summed = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def forward(self, inputs: list[torch.Tensor] | tuple[torch.Tensor, ...]):
        if len(inputs) < 8:
            raise ValueError(
                "LCF_BERT expects 8 input tensors: concat ids, segment ids, concat mask, "
                "local ids, local mask, aspect_begin, aspect_len, lcf_context_weight."
            )

        (
            concat_ids,
            concat_segments,
            concat_mask,
            local_ids,
            local_mask,
            _aspect_begin,
            _aspect_len,
            lcf_context_weight,
        ) = inputs[:8]

        concat_mask = concat_mask.long()
        local_mask = local_mask.long()

        spc_out = self._bert_forward(
            input_ids=concat_ids,
            attention_mask=concat_mask,
            token_type_ids=concat_segments,
        )
        spc_out = self.dropout(spc_out)
        spc_out = spc_out * concat_mask.unsqueeze(-1).to(spc_out.dtype)

        local_out = self._bert_forward(
            input_ids=local_ids,
            attention_mask=local_mask,
            token_type_ids=None,
        )
        local_out = self.dropout(local_out)
        local_out = local_out * local_mask.unsqueeze(-1).to(local_out.dtype)

        lcf_w = lcf_context_weight.to(dtype=local_out.dtype).unsqueeze(-1)
        local_out = local_out * lcf_w

        fused = self.linear_cat(torch.cat([local_out, spc_out], dim=-1))
        fused = self.bert_SA(fused, concat_mask)
        pooled = self._masked_mean(fused, concat_mask)

        return self.sentiment_head(pooled), self.aspect_head(pooled)
