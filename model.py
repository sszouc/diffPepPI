from __future__ import annotations

import math
import torch
from torch import nn

from config import (
    D_MODEL,
    NUM_LAYERS,
    NHEAD,
    DROPOUT,
    VOCAB_SIZE,
    OUT_CLASSES,
    PAD_IDX,
    PEPTIDE_LENGTH,
    MEMORY_LENGTH,
    EMBEDDING_DIM,
)

class TimeEmbedding(nn.Module):

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.d_model % 2 == 1:

            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return self.proj(emb)

class ConditionalDecoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_IDX)

        self.pos_emb = nn.Embedding(PEPTIDE_LENGTH, D_MODEL)
        self.time_emb = TimeEmbedding(D_MODEL)
        self.cond_proj = nn.Linear(EMBEDDING_DIM, D_MODEL)
        self.mem_pos_emb = nn.Embedding(MEMORY_LENGTH, D_MODEL)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dropout=DROPOUT,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=NUM_LAYERS)
        self.out_proj = nn.Linear(D_MODEL, OUT_CLASSES)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x_t.shape

        tok_emb = self.token_emb(x_t)

        pos_ids = torch.arange(seq_len, device=x_t.device).unsqueeze(0)

        pos_emb = self.pos_emb(pos_ids)

        time_emb = self.time_emb(t).unsqueeze(1)

        tgt = tok_emb + pos_emb + time_emb

        tgt = tgt.transpose(0, 1)

        cond_proj = self.cond_proj(cond)

        memory = cond_proj.unsqueeze(0).repeat(MEMORY_LENGTH, 1, 1)

        mem_pos = self.mem_pos_emb(torch.arange(MEMORY_LENGTH, device=x_t.device)).unsqueeze(1)

        memory = memory + mem_pos

        pad_mask = x_t.eq(PAD_IDX)

        dec_out = self.decoder(tgt=tgt, memory=memory, tgt_key_padding_mask=pad_mask)

        dec_out = dec_out.transpose(0, 1)
        logits = self.out_proj(dec_out)
        return logits

class LengthPredictor(nn.Module):

    def __init__(self, hidden: int = D_MODEL) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, hidden),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, PEPTIDE_LENGTH),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)
