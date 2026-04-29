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


class LCF_BERT(nn.Module):
    """LCF-BERT (Zeng et al., 2019) — single-task ABSC.

    Local branch (text only):
        BERT_local  ->  CDM/CDW (precomputed)  ->  MH Self-Attention
    Global branch (text + aspect, BERT-SPC):
        BERT_global  ->  MH Self-Attention
    Fusion / Output (FIL):
        Concatenate (⊕)  ->  Linear  ->  MH Self-Attention
        ->  pool [CLS]  ->  sentiment softmax (Equation 25–26)

    BERT replaces the embedding + Pre-Feature Extractor (PFE) layer entirely
    when used in LCF-BERT (paper §3.4 + Figure 3 caption); PCT belongs to
    LCF-GloVe and is intentionally absent here.
    """

    def __init__(self, bert: nn.Module, opt: SimpleNamespace):
        super().__init__()
        self.bert_spc = bert
        self.bert_local = bert
        self.opt = opt

        hidden_size = int(opt.bert_dim)
        num_heads = int(getattr(opt, "num_attention_heads", 12))
        dropout = float(opt.dropout)

        self.dropout = nn.Dropout(dropout)

        self.local_SA = MHSelfAttention(hidden_size, num_heads, dropout=dropout)
        self.global_SA = MHSelfAttention(hidden_size, num_heads, dropout=dropout)

        self.linear_cat = nn.Linear(hidden_size * 2, hidden_size)
        self.bert_SA = MHSelfAttention(hidden_size, num_heads, dropout=dropout)

        self.sentiment_head = nn.Linear(hidden_size, int(opt.polarities_dim))

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

    def forward(self, inputs: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
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

        # Local CDM/CDW (precomputed weights) then MH Self-Attention
        lcf_w = lcf_context_weight.to(dtype=local_out.dtype).unsqueeze(-1)
        local_out = local_out * lcf_w
        local_out = self.local_SA(local_out, local_mask)

        # Global MH Self-Attention
        global_out = self.global_SA(global_out, concat_mask)

        # Feature Interactive Learning: concat -> linear -> MHSA
        fused = self.linear_cat(torch.cat([local_out, global_out], dim=-1))
        fused = self.bert_SA(fused, concat_mask)

        # Output layer: pool the [CLS] (first-token) hidden state (paper eq 25)
        pooled = fused[:, 0, :]
        return self.sentiment_head(pooled)
