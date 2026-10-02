#model.py
"""去噪网络：条件 Transformer 解码器（ConditionalDecoder）+ 时间步嵌入（TimeEmbedding）。

整体是“编码器-解码器”里的解码器部分：蛋白条件（ESM2 嵌入，离线算好、训练时不更新）
被投影成若干 memory token，作为 Transformer 解码器交叉注意力的 Key/Value；
肽序列（含 MASK）作为目标序列，配合时间步嵌入，输出每个位置 20 种氨基酸的 logits。

张量形状约定：
    x_t   : (B, L)                 long     加噪后的肽索引（0..19 氨基酸，20 = PAD，21 = MASK）
    t     : (B,)                   long     每个样本的扩散时间步
    cond  : (B, EMBEDDING_DIM)     float32  蛋白条件向量
    logits: (B, L, OUT_CLASSES=20) float32  每个位置对 20 种氨基酸的分数
"""
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
    """把整数时间步 t 变成 d_model 维向量（正弦位置编码 + 两层 MLP）。

    与 DDPM 里常用的做法一致：先用不同频率的 sin/cos 把标量 t 展开成 d_model 维
    连续特征（这样相近的 t 得到相近的编码），再用一个小 MLP 投影学习。
    输出会加到 token 嵌入上，让网络知道“现在噪声有多重”。

    形状：输入 (B,) → 输出 (B, d_model)
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) → (B, d_model)。

        细节：
          · half_dim = d_model // 2，用一半维度放 sin、一半放 cos；
          · emb 先算频率因子 exp(-i * ln(10000)/(half_dim-1))（几何级数衰减的频率）；
          · t 乘上频率再取 sin/cos，等价于 Transformer 的绝对位置编码套路，
            只是这里的“位置”换成了扩散时间步。
        """
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)                          # 频率的指数底
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)  # (half_dim,)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)                  # (B, half_dim) 广播相乘
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)         # (B, 2*half_dim)
        if self.d_model % 2 == 1:
            # d_model 为奇数时补一列 0，保证通道数正好等于 d_model
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return self.proj(emb)                                           # (B, d_model)


class ConditionalDecoder(nn.Module):
    """条件去噪网络主体。

    组成部分（全部在 __init__ 里建好）：
        token_emb   : token 嵌入表（VOCAB_SIZE = 24 × d_model，含 20 氨基酸 + EOS + PAD + MASK + X；
                      PAD 行的梯度被 padding_idx 置零）
        pos_emb     : 序列位置嵌入（PEPTIDE_LENGTH = 51 × d_model，可学习）
        time_emb    : 时间步嵌入（TimeEmbedding）
        cond_proj   : 把 ESM2 蛋白向量投影到 d_model
        mem_pos_emb : memory token 的位置嵌入（16 × d_model，可学习）
        decoder     : 4 层 nn.TransformerDecoderLayer（每层含自注意力 + 交叉注意力 + FFN）
        out_proj    : 输出头，d_model → OUT_CLASSES = 21 类（20 种氨基酸 + <EOS>）

    关于 memory token（**容易误解的地方**）：
        代码是 cond_proj(cond) 得到 1 个 d_model 向量后，用 .repeat(MEMORY_LENGTH,1,1)
        复制成 16 份，再各自加上**不同的**可学习位置嵌入。
        也就是说 16 个 memory token 的差异完全来自 mem_pos_emb；
        （如果想让它们各自编码蛋白的不同区域，需要把 cond_proj 换成
          Linear(EMBEDDING_DIM, MEMORY_LENGTH * D_MODEL) 再 reshape 成 16 个 token。）
    """

    def __init__(self) -> None:
        super().__init__()
        # padding_idx=PAD_IDX：PAD 的嵌入向量固定为全零且不参与梯度更新
        # 生成的是每个氨基酸字符的初始特征
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_IDX)
        # 生成的是每个肽的初始特征
        self.pos_emb = nn.Embedding(PEPTIDE_LENGTH, D_MODEL)
        self.time_emb = TimeEmbedding(D_MODEL)
        self.cond_proj = nn.Linear(EMBEDDING_DIM, D_MODEL)
        self.mem_pos_emb = nn.Embedding(MEMORY_LENGTH, D_MODEL)
        # 说明：早期版本这里还有一个"长度嵌入"（len_emb），把 "这条肽有多长" 也当条件喂进来。
        # 但实测它没被用上：训练输入 x_t 是从真实 x0 加噪来的，位置 L 上是 <EOS>、
        # 之后全是 <PAD>（PAD 位永不掩码，还会通过 padding mask 告诉自注意力哪些位是填充），
        # 模型直接从输入就能读出真实长度 → 喂进去的长度条件成了冗余信息
        #（验证集上把条件从"预测长度"换成"真实长度"，损失只差 0.002）。
        # 因此这一版把长度从网络输入里删掉了：预测长度只在生成时用来**定采样窗口**。

        # batch_first=False → 张量形状按 (序列长度, batch, 维度) 组织，
        # 因此 forward 里要把 (B, L, D) 转置成 (L, B, D)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dropout=DROPOUT,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=NUM_LAYERS)
        self.out_proj = nn.Linear(D_MODEL, OUT_CLASSES)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """前向：为每个位置预测“它原本是哪个 token”的 21 类分布（即估计 x_0 的分布）。

        ========================= 1) 形状总览 =========================
        记号（本项目取值）：
            B = 一次并行处理的肽条数 = config.BATCH_SIZE      = 64
            L = 序列位置数           = PEPTIDE_LENGTH        = 51（最长肽 50 + EOS）
            D = 隐层维度             = D_MODEL               = 256
            E = 蛋白嵌入维度         = EMBEDDING_DIM         = 1280
            M = 记忆 token 数        = MEMORY_LENGTH         = 16

        输入（注意：没有"长度"这一项——长度只在生成时用来定窗口，见 diffusion.p_sample_loop）：
            x_t    : (B, L)  long    加噪后的序列索引（0..19 氨基酸 / 20 EOS / 21 PAD / 22 MASK / 23 X）
            t      : (B,)    long    每条样本各自的扩散时间步
            cond   : (B, E)  float32 蛋白条件向量，**一行对应一条肽**
        中间：
            tgt    : (L, B, D)       解码器的“目标序列”（肽），batch 在第 1 维
            memory : (M, B, D)       解码器的“记忆”（蛋白条件），batch 同样在第 1 维
        输出：
            logits : (B, L, 21)      每个位置对 21 个类别（20 种氨基酸 + <EOS>）的打分，
                                     未过 softmax；类别下标 0..20 与 token 下标 0..20 对齐

        关键点：**每个张量的第 0 维（或第 1 维）都带着 batch**——
        一次前向就同时算完 B 条肽，这 B 条肽的蛋白条件也都在 cond 里，
        所以不存在“一条一条算”的循环。

        ========================= 2) 为什么进解码器前要转置 =========================
        nn.TransformerDecoder 默认 batch_first=False，要求输入是
            (序列长度, batch, 特征) = (L, B, D)
        而 data_utils 给出的数据是“batch 在前”的 (B, L, ...)，
        因此进解码器前 transpose(0, 1)，出解码器后再 transpose(0, 1) 转回来。

        ========================= 3) 三条“相加”的信息 =========================
            tok_emb  : 当前位置是什么 token（可能就是 <MASK>）
            pos_emb  : 位置信息（这是第几个残基）
            time_emb : 噪声程度（当前扩散时间步 t）
        三者都是 D 维，逐元素相加得到解码器的输入。相加能成立全靠广播：
            tok_emb  (B, L, D)
            pos_emb  (1, L, D)  → 沿 batch 维复制 B 份（所有样本共用同一套位置向量）
            time_emb (B, 1, D)  → 沿序列维复制 L 份（同一条样本的所有位置共用同一时刻）
            ---------------------------------------------------------------
            结果      (B, L, D)

        ========================= 4) 掩码 =========================
            tgt_key_padding_mask = pad_mask，形状 (B, L)，True 表示“该位置是 PAD，
            算注意力时请忽略它”。它作用于**自注意力**：肽尾部补齐的 PAD 不参与计算，
            否则 PAD 会污染其它位置。
            这里不传 tgt_mask（因果掩码）：去噪任务是双向的，每个位置都要能看到整条序列，
            不能像语言模型那样只能看左边。
            也不传 memory_key_padding_mask：memory 全由蛋白条件生成，没有 padding。
        """
        batch_size, seq_len = x_t.shape                        # B, L（batch_size 只用于解包，后面没再用）
        # (B, L) → (B, L, D)：把每个离散 token 查表成 D 维向量（PAD 行恒为 0 向量）
        tok_emb = self.token_emb(x_t)
        # 位置 id 固定为 0..L-1；unsqueeze(0) 变成 (1, L)，让结果能在 batch 维广播
        pos_ids = torch.arange(seq_len, device=x_t.device).unsqueeze(0)
        # (1, L) → (1, L, D)：所有样本共享同一套可学习位置嵌入
        pos_emb = self.pos_emb(pos_ids)
        # (B,) → (B, D) → (B, 1, D)：每条样本一个时间向量，可在序列维广播
        time_emb = self.time_emb(t).unsqueeze(1)
        # 广播相加：tok_emb / time_emb 都是 (B,*,D) 级别，(1,L,D) 的 pos_emb 自动铺开
        tgt = tok_emb + pos_emb + time_emb                     # (B, L, D)

        # (B, L, D) → (L, B, D)：适配 batch_first=False 的解码器
        tgt = tgt.transpose(0, 1)

        # (B, E) → (B, D)：把 1280 维蛋白嵌入投影到隐层维度
        cond_proj = self.cond_proj(cond)
        # (B, D) → (1, B, D) → (16, B, D)：沿新加的第 0 维复制 16 份作为记忆 token。
        # 此刻 16 行完全相同（repeat 是真实复制数据，不是 expand 的视图）
        memory = cond_proj.unsqueeze(0).repeat(MEMORY_LENGTH, 1, 1)
        # arange(16) → (16,) → Embedding → (16, D) → (16, 1, D)：
        # 给 16 个记忆槽各配一个可学习向量，**这才是 16 个 token 彼此唯一的差异来源**
        mem_pos = self.mem_pos_emb(torch.arange(MEMORY_LENGTH, device=x_t.device)).unsqueeze(1)
        # (16, B, D) + (16, 1, D) → (16, B, D)：同一套记忆位置向量对所有样本广播
        memory = memory + mem_pos

        # (B, L) 的布尔张量：True = 该位置是 PAD，自注意力要屏蔽
        pad_mask = x_t.eq(PAD_IDX)
        # tgt 提供 Query（L 个肽位置），memory 提供 Key/Value（16 个记忆 token）；
        # 每个肽位置都会对 16 个记忆做一次加权求和，所以输出形状与输入 tgt 相同 (L, B, D)
        dec_out = self.decoder(tgt=tgt, memory=memory, tgt_key_padding_mask=pad_mask)
        # dec_out: (L, B, D)
        # 转回 batch 在前的布局，方便按 (B, L, 21) 与 x0 逐位置算交叉熵
        dec_out = dec_out.transpose(0, 1)                      # (B, L, D)
        logits = self.out_proj(dec_out)                        # (B, L, D) → (B, L, 21)
        return logits


class LengthPredictor(nn.Module):
    """长度预测器：只看蛋白嵌入，预测这条肽有多长（0..PEPTIDE_LENGTH-1 共 51 类）。

    为什么要单独一个模块：
        · 掩码扩散是并行、双向去噪的，模型没法"生成到 EOS 自然停"（这一点我们已经用
          实验证实：尾部 PAD 会成为捷径、EOS 自由采样又会过早终止）；
        · 所以把"多长"这件事拆出来单独预测。生成时先让本模块给出长度 k，
          再让扩散模型只去填 0..k-1 这 k 个位置——长度与内容解耦，训练也更稳。

    结构：3 层 MLP（1280 → 256 → 256 → 51），参数量约 0.4M。
    输入：cond (B, EMBEDDING_DIM)  蛋白嵌入
    输出：logits (B, PEPTIDE_LENGTH)  长度分类的打分（第 i 类 = 长度 i）

    训练目标：`F.cross_entropy(logits, true_length)`，与扩散损失加权求和一起反传。
    推理用法：把输出过 softmax 得到 p(k|蛋白)，再 **按该分布采样**得到 k：
        `k = torch.multinomial(torch.softmax(logits, dim=-1), 1)`
        （评估/生成脚本用的就是这种 τ=1 采样：argmax 会系统性偏向短肽，
          实测它与真实长度分布的 TV 距离 0.28，采样后降到 0.11）。
    """

    def __init__(self, hidden: int = D_MODEL) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, hidden),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, PEPTIDE_LENGTH),      # 输出 51 类：长度 0..50
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """cond: (B, EMBEDDING_DIM) → logits: (B, PEPTIDE_LENGTH)。"""
        return self.net(cond)
