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
    """去噪网络：4 层 TransformerDecoder，条件用标准交叉注意力注入。

    Q 来自肽（51 个位置），K/V 是蛋白条件投影后的单个 memory token。
    """

    def __init__(self) -> None:
        super().__init__()

        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_IDX)

        self.pos_emb = nn.Embedding(PEPTIDE_LENGTH, D_MODEL)
        self.time_emb = TimeEmbedding(D_MODEL)
        self.cond_proj = nn.Linear(EMBEDDING_DIM, D_MODEL)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dropout=DROPOUT,
            batch_first=False,
        )
        # 单层解码器。包括自注意力，交叉注意力，MLP
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=NUM_LAYERS)
        self.out_proj = nn.Linear(D_MODEL, OUT_CLASSES)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:

        # x_t是时间步t下的肽，cond是蛋白质特征
        # x_t形状是(B, 51)
        # t的形状是(B, 1)
        batch_size, seq_len = x_t.shape
        # seq_len是51，没有batch
        tok_emb = self.token_emb(x_t)
        # 输入肽，线性变换得到一个256维的向量
        # tok_emb 形状是(B, 51, 256)，给每个位置的氨基酸分配一个256维度的向量
        pos_ids = torch.arange(seq_len, device=x_t.device).unsqueeze(0)
        # 生成1到51的等差数列，也就是位置索引。形状是(1,51)，
        pos_emb = self.pos_emb(pos_ids)
        # 输入位置索引，线性变换给每个位置一个256维的向量。形状是(1,51，256)
        time_emb = self.time_emb(t).unsqueeze(1)
        # 输入此时的时间t，得到一个256维度的向量，形状是(B, 256)，然后还要加一个维度(B, 1 ,256)
        tgt = tok_emb + pos_emb + time_emb

        tgt = tgt.transpose(0, 1)
        # tgt (51,batch256)

        cond_proj = self.cond_proj(cond)
        # 把蛋白质从1280维线性变换成256维向量 (batch,256)
        memory = cond_proj.unsqueeze(0)
        # 变成 (1,batch,256)
        pad_mask = x_t.eq(PAD_IDX)
        dec_out = self.decoder(tgt=tgt, memory=memory, tgt_key_padding_mask=pad_mask)
        dec_out = dec_out.transpose(0, 1)
        logits = self.out_proj(dec_out)
        #输出(1,batch,21)，这里的21是20种氨基酸加上<EOS>
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
