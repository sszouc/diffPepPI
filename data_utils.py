#data_utils.py
"""数据层：把 FASTA 文本 + 预计算蛋白嵌入，组装成训练 / 验证 DataLoader。

数据流：
    Train_Sequences.fasta
        └─ parse_fasta_pairs() ──> [(蛋白序列, 肽序列), ...]
    protein_esm.pkl {蛋白序列: numpy(EMBEDDING_DIM,)}
        └─ PeptideDataset      ──> (cond, x0, true_len) 三元组
                                    └─ get_dataloaders() 按 9:1 划分训练 / 验证

张量形状约定（与 model.py / diffusion.py 一致）：
    cond     : (B, EMBEDDING_DIM)   float32  蛋白条件向量（ESM2 嵌入）
    x0       : (B, PEPTIDE_LENGTH)  long     干净肽索引：0..19 = 氨基酸，20 = PAD
    true_len : (B,)                 long     肽的真实长度（本项目只记录，未参与损失屏蔽）
"""
from __future__ import annotations

import os
import pickle
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from config import (
    AMINO_ACIDS,
    PAD_IDX,
    PEPTIDE_LENGTH,
    DATA_PATH,
    EMBEDDING_PATH,
    SEED,
    ONLY_POSITIVE,
    EOS_IDX,
    X_IDX,
)

# 氨基酸字符 → 类别索引：{'A': 0, 'C': 1, ..., 'Y': 19}
AMINO_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def set_seed(seed: int) -> None:
    """固定 python / numpy / torch（含全部 GPU）的随机种子，保证实验可复现。

    注意：这里没有设置 cudnn.deterministic，GPU 上算子累加顺序仍可能带来微小的
    浮点差异；但数据划分与随机时间步 t 的采样结果是确定的。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_fasta_pairs(fasta_path: str, only_positive: bool = False) -> List[Tuple[str, str]]:
    """读取类 FASTA 文件，返回 [(蛋白序列, 肽序列), ...]。

    文件格式（每条记录 3 行）：
        >positive_1                      ← 以 '>' 开头的记录头，用来说明这条记录是正样本还是负样本
        Peptide: XYVYNTRSGWRWYT          ← 肽序列（取第一个冒号之后的内容）
        Protein: MHHHHHHSSGVDLG...       ← 蛋白序列

    参数：
        only_positive : True 时只收下以 '>positive' 开头的记录（正样本），
                        负样本（'>negative'）会被跳过。训练入口按 config.ONLY_POSITIVE 传入。

    实现要点与注意事项：
      1. 记录头**只**用来判断正负（当 only_positive=True 时）和分隔记录；
         当前 Train_Sequences.fasta 里两类各 8622 条，合计 17244 条。
      2. Peptide: 与 Protein: 的先后顺序无所谓（实际文件里 Peptide 在前），
         只有当两个字段都取到时，才会在读到下一条记录头时把该对追加进结果。
      3. 文件末尾没有额外的 '>' 收尾，所以循环结束后必须再判断一次，
         否则最后一条记录会被丢掉。
      4. 任何缺胳膊少腿的记录（没有 Peptide 或没有 Protein）会被静默跳过。
    """
    pairs: List[Tuple[str, str]] = []
    current_protein = None
    current_peptide = None
    keep_record = True                                # 当前这条记录是否要收下（由记录头决定）

    with open(fasta_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:                                  # 跳过空行
                continue
            if line.startswith(">"):                      # 新记录开始 → 结算上一条
                if keep_record and current_protein is not None and current_peptide is not None:
                    pairs.append((current_protein, current_peptide))
                current_protein, current_peptide = None, None   # 清空状态，准备读新记录
                # 只筛正样本时，记录头必须以 ">positive" 开头（注意有个别头后面还跟了多余字符）
                keep_record = (not only_positive) or line.startswith(">positive")
                continue
            if line.startswith("Protein:"):
                # split(..., 1)：即使序列里再出现 "Protein:" 也不会被切碎
                current_protein = line.split("Protein:", 1)[1].strip()
            elif line.startswith("Peptide:"):
                current_peptide = line.split("Peptide:", 1)[1].strip()

    if keep_record and current_protein is not None and current_peptide is not None:   # 补上最后一条记录
        pairs.append((current_protein, current_peptide))

    return pairs


def encode_peptide(seq: str, length: int = PEPTIDE_LENGTH) -> Tuple[List[int], int]:
    """把一条肽序列编码成“氨基酸… + <EOS> + <PAD>…”的定长索引列表。

    参数：
        seq    : 肽字符串（大小写均可，内部先 strip 再转大写）
        length : 目标长度，默认 PEPTIDE_LENGTH = 51（数据最长 50 个残基 + 1 个 EOS）

    返回：
        indices     : 长度恰好为 length 的 list[int]
        true_length : 肽的真实长度（不含 EOS）= min(len(seq), length - 1)

    规则（本版本已改成变长友好的形式）：
        · 肽本体最多保留 length - 1 个残基（要给末尾的 <EOS> 留一个位置），
          超长时仍是保留 N 端、截掉 C 端；由于 L_max = 51 > 数据最长 50，
          实际上没有任何肽再被截断；
        · 肽的最后紧跟一个 <EOS>（索引 20）——它是网络的第 21 个输出类别，
          训练时用它学会“在这个蛋白条件下肽到第几位结束”，采样时解码到它即停止；
        · 其余位置补 <PAD>（索引 21），保证所有样本形状统一；
        · 未知字符（数据里的 'X'）编码成专门的 <X>（索引 23），
          不再被静默当成 'A'；<X> 只作为输入可见，训练时会从损失里排除
          （见 train.py 里把 X 位置映射成 PAD 的那一步）。
    """
    seq = seq.strip().upper()
    true_length = min(len(seq), length - 1)          # 留一个位置给 EOS
    indices = []
    for i in range(true_length):
        aa = seq[i]
        indices.append(AMINO_TO_IDX.get(aa, X_IDX))  # 未收录字符 → <X>（未知），而不是 'A'
    indices.append(EOS_IDX)                          # 肽结束标记
    if len(indices) < length:
        indices.extend([PAD_IDX] * (length - len(indices)))
    return indices, true_length


def decode_peptide(indices, max_len: int | None = None) -> str:
    """把索引序列翻译回肽字符串（train / evaluate / generate 共用这一份实现）。

    规则：
        · 只保留 0..19（标准氨基酸），<PAD> / <MASK> / <X> 都会被跳过；
        · 遇到 <EOS>（索引 20）也停止（正常情况下 EOS 出现在位置 k，见下）；
        · max_len 不为 None 时只看前 max_len 个位置。

    参数：
        indices : list[int] 或 torch.Tensor（一维）都支持
        max_len : 期望的肽长度 k。本项目现在的流程是"先预测长度 k，再生成 k 个残基"，
                  所以生成阶段一律传 max_len=k：即使模型在 k 之前偶然放了 EOS，
                  输出长度也正好是 k（长度由预测器决定，不由 EOS 决定）。
                  评估参考肽（FASTA 里的原始字符串）不需要它，保持默认 None。
    """
    if hasattr(indices, "tolist"):
        indices = indices.tolist()
    if max_len is not None:
        indices = indices[: int(max_len)]
    chars = []
    for idx in indices:
        if idx == EOS_IDX:
            break
        if 0 <= idx < len(AMINO_ACIDS):
            chars.append(AMINO_ACIDS[idx])
    return "".join(chars)


class PeptideDataset(Dataset):
    """(蛋白条件, 干净肽索引, 真实长度) 数据集。

    __init__ 时一次性把全部记录和对应的蛋白嵌入读进内存
    （嵌入约 5688 × 1280 × 4B ≈ 29 MB，肽索引可忽略），训练阶段不再做磁盘 IO；
    代价是样本在 DataLoader 多进程下会被反复 pickle，因此 num_workers 保持 0 更划算。
    """

    def __init__(self, fasta_path: str, embedding_path: str) -> None:
        # ---- 先做文件存在性检查，报错信息里带上路径，便于定位配置问题 ----
        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(
                f"Embedding file not found: {embedding_path}. "
                "Please precompute ESM2 embeddings first."
            )

        # 只取正样本（config.ONLY_POSITIVE=True）：FASTA 里的 >negative_* 记录不参与训练，
        # 否则模型会去拟合“结合肽 + 非结合肽”的混合分布
        pairs = parse_fasta_pairs(fasta_path, only_positive=ONLY_POSITIVE)

        # 嵌入文件是 {蛋白序列: numpy 数组} 的普通 pickle；
        with open(embedding_path, "rb") as f:
            embed_dict: Dict[str, np.ndarray] = pickle.load(f)

        self.samples: List[Tuple[np.ndarray, List[int], int]] = []
        missing = 0
        for protein_seq, peptide_seq in pairs:
            if protein_seq not in embed_dict:          # 没算过嵌入的蛋白 → 丢弃该样本
                missing += 1
                continue
            embedding = embed_dict[protein_seq]        # (EMBEDDING_DIM,) float32
            indices, true_length = encode_peptide(peptide_seq)
            self.samples.append((embedding, indices, true_length))
            # data每一条记录的格式都是：蛋白质的特征向量，其结合的肽的表示（不是字母，而是字母自己对应的数字），肽段的长度

        if missing > 0:
            # 正常情况下应为 0；出现说明 FASTA 与嵌入文件不是同一批数据
            print(f"Warning: {missing} proteins missing embeddings; skipped.")

    def __len__(self) -> int:
        """样本总数（只取正样本时 = 8622 条，减去缺嵌入被跳过的）。"""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """取一个样本，返回 (cond, x0, true_len)。

        这里把 numpy / list 转成 tensor：嵌入转 float32；肽索引与长度用 long。
        长度信息保留下来，方便以后做变长生成（EOS）之类的改造。
        """
        embedding, indices, true_length = self.samples[idx]
        embedding_t = torch.tensor(embedding, dtype=torch.float32)   # (EMBEDDING_DIM,)
        indices_t = torch.tensor(indices, dtype=torch.long)          # (PEPTIDE_LENGTH,)
        length_t = torch.tensor(true_length, dtype=torch.long)       # 0 维张量（标量）
        return embedding_t, indices_t, length_t


def get_dataloaders(
    batch_size: int,
    split_ratio: float = 0.9,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """构建 (训练 DataLoader, 验证 DataLoader)。

    参数：
        batch_size  : 每个 batch 的样本数（本项目用 config.BATCH_SIZE = 64）
        split_ratio : 训练集占比，默认 0.9（90% 训练 / 10% 验证）
        num_workers : DataLoader 子进程数，默认 0（数据都在内存里，多进程收益很小）

    细节：
        · 先 set_seed(SEED) 再建带种子的 Generator，保证每次划分结果完全一致；
        · random_split 用该 generator 打乱后切分：训练集 shuffle=True，
          验证集 shuffle=False（验证损失是按 batch 累加求和，顺序不影响结果）；
        · 样本数不能整除时 train_len 向下取整，余数归入验证集
          （17244 条 → 训练 15519 / 验证 1725）。
    """
    set_seed(SEED)
    dataset = PeptideDataset(DATA_PATH, EMBEDDING_PATH)
    train_len = int(len(dataset) * split_ratio)
    val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader
