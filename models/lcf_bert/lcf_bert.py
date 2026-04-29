from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch
import torch.nn as nn


class MHSelfAttention(nn.Module):
    """Multi-Head Self-Attention block (paper figure: 'MH Self-Attention')."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            for h in (8, 6, 4, 2, 1):
                if hidden_size % h == 0:
                    num_heads = h
                    break
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        kp_mask = None
        if attention_mask is not None:
            kp_mask = ~attention_mask.bool()
        out, _ = self.mha(x, x, x, key_padding_mask=kp_mask, need_weights=False)
        out = torch.nan_to_num(out, nan=0.0)
        if attention_mask is not None:
            out = out * attention_mask.unsqueeze(-1).to(out.dtype)
        return self.tanh(out)


class PointwiseConv(nn.Module):
    """Point-wise Convolutional Transformation (paper figure: 'PCT').

    Two 1x1 convolutions with ReLU between them, applied position-wise.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        h = self.act(self.conv1(h))
        h = self.conv2(h)
        return h.transpose(1, 2)


class LCF_BERT(nn.Module):
    """LCF-BERT replicating the architecture diagram in paper.PNG.

    Local branch (text only):
        BERT [Embedding + Pre-feature Extractor]  ->  PCT
        ->  CDM/CDW  ->  MH Self-Attention            (Feature Extractor)

    Global branch (text + aspect, BERT-SPC):
        BERT [Embedding + Pre-feature Extractor]  ->  PCT
        ->  MH Self-Attention                          (Feature Extractor)

    Fusion / Output:
        Concatenate (⊕)  ->  Linear  ->  MH Self-Attention   (FILL)
        ->  pool  ->  sentiment / aspect heads               (Output Layer)

    Input contract preserved (8 tensors): concat_ids, concat_segments,
    concat_mask, local_ids, local_mask, aspect_begin, aspect_len,
    lcf_context_weight (PyABSA CDM/CDW precomputed in the dataset).
    """

    def __init__(self, bert: nn.Module, opt: SimpleNamespace):
        super().__init__()
        self.bert_spc = bert
        self.bert_local = bert
        self.opt = opt

        hidden_size = int(opt.bert_dim)
        num_heads = int(getattr(opt, "num_attention_heads", 8))
        dropout = float(opt.dropout)

        self.dropout = nn.Dropout(dropout)

        self.pct_local = PointwiseConv(hidden_size)
        self.pct_global = PointwiseConv(hidden_size)

        self.local_SA = MHSelfAttention(hidden_size, num_heads, dropout=dropout)
        self.global_SA = MHSelfAttention(hidden_size, num_heads, dropout=dropout)

        self.linear_cat = nn.Linear(hidden_size * 2, hidden_size)

        self.bert_SA = MHSelfAttention(hidden_size, num_heads, dropout=dropout)

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

        # === BERT-shared layer alternative: Embedding + Pre-feature Extractor ===
        global_out = self._bert_forward(
            input_ids=concat_ids,
            attention_mask=concat_mask,
            token_type_ids=concat_segments,
        )
        global_out = self.dropout(global_out)
        global_out = global_out * concat_mask.unsqueeze(-1).to(global_out.dtype)

        local_out = self._bert_forward(
            input_ids=local_ids,
            attention_mask=local_mask,
            token_type_ids=None,
        )
        local_out = self.dropout(local_out)
        local_out = local_out * local_mask.unsqueeze(-1).to(local_out.dtype)

        # === PCT (Point-wise Convolutional Transformation) ===
        local_out = self.pct_local(local_out) * local_mask.unsqueeze(-1).to(local_out.dtype)
        global_out = self.pct_global(global_out) * concat_mask.unsqueeze(-1).to(global_out.dtype)

        # === Feature Extractor ===
        # Local: CDM/CDW (precomputed lcf_context_weight) -> MH Self-Attention
        lcf_w = lcf_context_weight.to(dtype=local_out.dtype).unsqueeze(-1)
        local_out = local_out * lcf_w
        local_out = self.local_SA(local_out, local_mask)

        # Global: MH Self-Attention
        global_out = self.global_SA(global_out, concat_mask)

        # === Concatenate (⊕) and project back to hidden_size ===
        fused = self.linear_cat(torch.cat([local_out, global_out], dim=-1))

        # === Feature Interactive Learning Layer ===
        fused = self.bert_SA(fused, concat_mask)

        # === Output Layer: pool ===
        pooled = self._masked_mean(fused, concat_mask)
        return self.sentiment_head(pooled), self.aspect_head(pooled)
