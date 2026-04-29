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

    Expected input list:
      0 concat_bert_indices          LongTensor [B, L]
      1 concat_segments_indices      LongTensor [B, L]
      2 concat_attention_mask        LongTensor [B, L]
      3 text_local_indices           LongTensor [B, L]
      4 text_local_attention_mask    LongTensor [B, L]
      5 aspect_begin                 LongTensor [B]
      6 aspect_len                   LongTensor [B]

    Returns:
      sentiment_logits, aspect_logits
    """

    def __init__(self, bert: nn.Module, opt: SimpleNamespace):
        super().__init__()
        self.bert_spc = bert
        self.bert_local = bert  # shared encoder, as in many compact LCF-BERT repos
        self.opt = opt

        hidden_size = int(getattr(opt, "bert_dim", getattr(bert.config, "hidden_size", 768)))
        self.dropout = nn.Dropout(float(getattr(opt, "dropout", 0.1)))
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

    def _local_context_weight(
        self,
        attention_mask: torch.Tensor,
        aspect_begin: torch.Tensor,
        aspect_len: torch.Tensor,
    ) -> torch.Tensor:
        """Build CDM/CDW weights using precomputed aspect span positions.

        If an aspect span is not found in the text, the sample keeps weight 1 on
        valid tokens. This is important for category labels such as food/service
        that may not appear literally in the sentence.
        """
        batch_size, seq_len = attention_mask.shape
        weights = torch.ones(
            batch_size,
            seq_len,
            dtype=torch.float32,
            device=attention_mask.device,
        )
        focus = str(getattr(self.opt, "local_context_focus", "cdw")).lower()
        srd = int(getattr(self.opt, "SRD", 3))

        for b in range(batch_size):
            valid_len = int(attention_mask[b].sum().item())
            begin = int(aspect_begin[b].item())
            length = int(aspect_len[b].item())
            if valid_len <= 0:
                continue
            if begin < 0 or length <= 0:
                # Aspect string not found in text. Keep all valid tokens.
                weights[b, valid_len:] = 0.0
                continue

            end = min(seq_len, begin + length)
            if focus == "cdm":
                left = max(0, begin - srd)
                right = min(seq_len, end + srd)
                weights[b, :left] = 0.0
                weights[b, right:] = 0.0
            elif focus == "cdw":
                for i in range(valid_len):
                    if begin <= i < end:
                        dist = 0.0
                    else:
                        dist = float(min(abs(i - begin), abs(i - (end - 1))))
                    if dist > srd:
                        weights[b, i] = max(0.0, 1.0 - (dist - srd) / max(float(valid_len), 1.0))
                weights[b, valid_len:] = 0.0
            else:
                weights[b, valid_len:] = 0.0

        return weights.unsqueeze(-1)

    @staticmethod
    def _masked_mean(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        summed = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def forward(self, inputs: list[torch.Tensor] | tuple[torch.Tensor, ...]):
        if len(inputs) < 7:
            raise ValueError(
                "LCF_BERT expects 7 input tensors: concat ids, segment ids, concat mask, "
                "local ids, local mask, aspect_begin, aspect_len."
            )

        (
            concat_ids,
            concat_segments,
            concat_mask,
            local_ids,
            local_mask,
            aspect_begin,
            aspect_len,
        ) = inputs[:7]

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

        context_weight = self._local_context_weight(local_mask, aspect_begin, aspect_len).to(local_out.dtype)
        local_out = local_out * context_weight

        fused = self.linear_cat(torch.cat([local_out, spc_out], dim=-1))
        fused = self.bert_SA(fused, concat_mask)
        pooled = self._masked_mean(fused, concat_mask)

        return self.sentiment_head(pooled), self.aspect_head(pooled)
