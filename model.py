"""去噪网络与长度分支。

条件是 `struct_encoder` 预先算好的**变长 memory**（全局 token + 逐残基结构表征）：
  - 肽的每个位置可以 attend 到蛋白的任意残基（交叉注意力 K/V 长度 = N+1）
  - 结构编码每个 batch 只做一次，扩散的 100 步复用同一份 memory
"""
from __future__ import annotations

import math

import torch
from torch import nn

from config import (
    D_MODEL, NUM_LAYERS, NHEAD, DROPOUT, VOCAB_SIZE, OUT_CLASSES,
    PAD_IDX, PEPTIDE_LENGTH, EMBEDDING_DIM, FFN_DIM, LEN_HIDDEN,
)


class TimeEmbedding(nn.Module):
    """正弦时间嵌入 + 两层 MLP。"""

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
    """4 层 TransformerDecoder，交叉注意力消费结构编码器产出的 memory。

    输入
      x_t           (B, 51)      加噪后的肽
      t             (B,)         时间步
      memory        (B, L, 256)  结构编码器输出（全局 token + 逐残基）
      memory_mask   (B, L)       布尔，True 表示该 memory 位置是 padding
    """

    def __init__(self) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_IDX)
        self.pos_emb = nn.Embedding(PEPTIDE_LENGTH, D_MODEL)
        self.time_emb = TimeEmbedding(D_MODEL)

        layer = nn.TransformerDecoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=FFN_DIM,
            dropout=DROPOUT,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=NUM_LAYERS)
        self.out_proj = nn.Linear(D_MODEL, OUT_CLASSES)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                memory: torch.Tensor, memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L = x_t.shape
        tgt = (self.token_emb(x_t)
               + self.pos_emb(torch.arange(L, device=x_t.device).unsqueeze(0))
               + self.time_emb(t).unsqueeze(1)).transpose(0, 1)      # (L,B,D)

        dec_out = self.decoder(
            tgt=tgt,
            memory=memory.transpose(0, 1),                           # (L_mem,B,D)
            tgt_key_padding_mask=x_t.eq(PAD_IDX),
            memory_key_padding_mask=memory_mask,
        )
        return self.out_proj(dec_out.transpose(0, 1))


class LengthPredictor(nn.Module):
    """从全局 ESM 向量预测肽长分布（51 类）。

    隐层宽度 LEN_HIDDEN=256。曾试过压到 64（省 31.7 万参数）：NLL 基本不变，
    但采样开始输出越界长度（3415 条里出现 0/35/41/49），故保留 256。
    这个子问题相对边际基线只能提升约 0.07 nats，也不宜再加大。
    """

    def __init__(self, hidden: int = LEN_HIDDEN) -> None:
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
