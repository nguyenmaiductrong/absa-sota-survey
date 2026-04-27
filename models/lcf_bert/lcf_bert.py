import torch
import torch.nn as nn
import numpy as np

from transformers import BertModel


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, max_seq_len: int, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.device = device
        self.query  = nn.Linear(hidden_size, hidden_size)
        self.key    = nn.Linear(hidden_size, hidden_size)
        self.value  = nn.Linear(hidden_size, hidden_size)
        self.tanh   = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.hidden_size ** 0.5)
        attn   = torch.softmax(scores, dim=-1)
        out    = torch.bmm(attn, v)
        return self.tanh(out)


class LCF_BERT(nn.Module):
    """LCF-BERT with dual output heads (aspect category + sentiment).

    Input tensors (all shape: batch × max_seq_len):
      0 - text_bert_indices      : [CLS] text [SEP] aspect [SEP]
      1 - bert_segments_ids      : token-type-id segment mask
      2 - text_local_indices     : [CLS] text [SEP]
      3 - aspect_indices         : [CLS] aspect [SEP]

    Outputs:
      (sentiment_logits, aspect_logits)  — each (batch, n_classes)
    """

    def __init__(self, bert: BertModel, opt):
        super().__init__()
        self.bert_spc   = bert
        self.bert_local = bert          # shared weights (single BERT)
        self.opt        = opt

        D = opt.bert_dim
        self.dropout    = nn.Dropout(opt.dropout)
        self.bert_SA    = SelfAttention(D, opt.max_seq_len, opt.device)
        self.linear_cat = nn.Linear(D * 2, D)

        self.sentiment_head = nn.Linear(D, opt.polarities_dim)
        self.aspect_head    = nn.Linear(D, opt.aspects_dim)

    # ── Local context focus helpers ──────────────────────────────────────────

    def _cdm_mask(self, text_local_indices: torch.Tensor,
                  aspect_indices: torch.Tensor) -> torch.Tensor:
        """Context-dependent masking (CDM)."""
        texts = text_local_indices.cpu().numpy()
        asps  = aspect_indices.cpu().numpy()
        B, L, D = texts.shape[0], self.opt.max_seq_len, self.opt.bert_dim
        mask = np.ones((B, L, D), dtype=np.float32)
        srd  = self.opt.SRD

        for bi in range(B):
            asp_len = int(np.count_nonzero(asps[bi])) - 2
            try:
                asp_begin = int(np.argwhere(texts[bi] == asps[bi][1])[0][0])
            except IndexError:
                continue
            start = max(0, asp_begin - srd)
            for i in range(start):
                mask[bi][i] = 0
            for j in range(asp_begin + asp_len + srd, L):
                mask[bi][j] = 0

        return torch.from_numpy(mask).to(self.opt.device)

    def _cdw_weight(self, text_local_indices: torch.Tensor,
                    aspect_indices: torch.Tensor) -> torch.Tensor:
        """Context-dependent weighting (CDW)."""
        texts = text_local_indices.cpu().numpy()
        asps  = aspect_indices.cpu().numpy()
        B, L, D = texts.shape[0], self.opt.max_seq_len, self.opt.bert_dim
        weight = np.ones((B, L, D), dtype=np.float32)
        srd    = self.opt.SRD

        for bi in range(B):
            asp_len = int(np.count_nonzero(asps[bi])) - 2
            try:
                asp_begin = int(np.argwhere(texts[bi] == asps[bi][1])[0][0])
                asp_center = (asp_begin * 2 + asp_len) / 2
            except IndexError:
                continue
            n_nonzero = int(np.count_nonzero(texts[bi]))
            for i in range(1, n_nonzero - 1):
                dist = abs(i - asp_center) + asp_len / 2
                if dist > srd:
                    weight[bi][i] *= 1 - (dist - srd) / n_nonzero

        return torch.from_numpy(weight).to(self.opt.device)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, inputs: list[torch.Tensor]):
        text_bert_indices, bert_segments_ids, text_local_indices, aspect_indices = inputs

        # SPC branch: [CLS] text [SEP] aspect [SEP]
        spc_out = self.bert_spc(
            text_bert_indices, token_type_ids=bert_segments_ids
        ).last_hidden_state
        spc_out = self.dropout(spc_out)

        # Local branch: [CLS] text [SEP]
        local_out = self.bert_local(text_local_indices).last_hidden_state
        local_out = self.dropout(local_out)

        lcf = self.opt.local_context_focus
        if lcf == "cdm":
            mask = self._cdm_mask(text_local_indices, aspect_indices)
            local_out = local_out * mask
        elif lcf == "cdw":
            w = self._cdw_weight(text_local_indices, aspect_indices)
            local_out = local_out * w

        # Concat → self-attention → mean pool
        fused = self.linear_cat(torch.cat([local_out, spc_out], dim=-1))
        fused = self.bert_SA(fused)
        pooled = fused.mean(dim=1)          # mean pooling over tokens

        return self.sentiment_head(pooled), self.aspect_head(pooled)
