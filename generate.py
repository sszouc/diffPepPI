#generate.py
"""交互式生成脚本：给一条蛋白序列，先生成它的肽长度，再按该长度采样若干候选肽。

流程（与 evaluate.py 完全一致）：
    cond = ESM2 嵌入(蛋白) → LengthPredictor 给出长度分布 p(k|蛋白)
    → **按 τ=1 的 softmax 采样**得到每条候选肽自己的长度 k
    → p_sample_loop 只采样位置 0..k-1，位置 k 放 <EOS>，其余 <PAD>
    → 解码出恰好 k 个残基（同一蛋白的多条候选肽长度通常不同）

用法（蛋白序列必须在 protein_esm.pkl 里已经算过嵌入，否则会抛 KeyError）：
    python generate.py --protein_seq MHHHHHHSSGVDLG...            # 默认生成 5 条
    python generate.py --protein_seq MHHH... --num_samples 20
    python generate.py --protein_seq MHHH... --checkpoint checkpoints/model_epoch_100.pt

与 evaluate.py 的区别：这里只处理**一个**蛋白（条件由命令行给出），
适合人工看某个目标蛋白的生成结果；evaluate.py 是批量跑全库统计指标。
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import torch

from config import (
    DEVICE,
    PEPTIDE_LENGTH,
    EMBEDDING_PATH,
    EMBEDDING_DIM,
)
from data_utils import decode_peptide          # 解码统一走 data_utils：按 max_len 截断
from diffusion import p_sample_loop
from model import ConditionalDecoder, LengthPredictor


def load_embedding_from_pkl(seq: str, pkl_path: str) -> np.ndarray:
    """按蛋白序列从嵌入 pickle 里取出对应向量；不存在则抛 KeyError。

    注意每次调用都会把整个 pickle 读一遍（本项目只用一次，可接受）。
    """
    with open(pkl_path, "rb") as f:
        embed_dict = pickle.load(f)
    if seq not in embed_dict:
        raise KeyError("Protein sequence not found in embedding file.")
    return embed_dict[seq]


def main() -> None:
    """命令行入口：取条件 → 载入权重 → 采样 → 打印结果。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
                        type=str,
                        default="checkpoints/model_epoch_100.pt")   # 与 config.EPOCHS=100 对应的最终权重
    parser.add_argument(
        "--protein_seq",
                        type=str,
                        default="MADFEDRVSDEEKVRIAAKFITHAPPGEFNEVFNDVRLLLNNDNLLREGAAHAFAQYNMDQFTPVKIEGYDDQVLITEHGDLGNGRFLDPRNKISFKFDHLRKEASDPQPEDTESALKQWRDACDSALRAYVKDHYPNGFCTVYGKSIDGQQTIIACIESHQFQPKNFWNGRWRSEWKFTITPPTAQVAAVLKIQVHYYEDGNVQLVSHKDIQDSVQVSSDVQTAKEFIKIIENAENEYQTAISENYQTMSDTTFKALRRQLPVTRTKIDWNKILSYKIGKEMQNA")                              # 必填：蛋白全长序列
    parser.add_argument(
        "--num_samples",
                        type=int,
                        default=5)                                  # 生成多少条候选肽
    args = parser.parse_args()

    # ---- 1. 条件准备：查嵌入 → 检查维度 → 加 batch 维并搬到设备 ----
    embedding = load_embedding_from_pkl(args.protein_seq, EMBEDDING_PATH)
    if embedding.shape[-1] != EMBEDDING_DIM:      # 防止嵌入文件与 config 的维度不一致
        raise ValueError(
            f"Embedding dim mismatch: expected {EMBEDDING_DIM}, got {embedding.shape[-1]}"
        )
    cond = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(DEVICE)   # (1, D)

    # ---- 2. 载入权重 ----
    # 权重文件里同时存了去噪网络与长度预测器（train.py 存的字典），两个都要加载
    # 注意：这里没有传 weights_only=True（evaluate.py 里有传），
    # 新版 PyTorch 默认值已经是 True，行为等价；但显式写出更稳妥
    model = ConditionalDecoder().to(DEVICE)
    length_predictor = LengthPredictor().to(DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt["decoder"])
    length_predictor.load_state_dict(ckpt["length_predictor"])
    model.eval()
    length_predictor.eval()

    # ---- 3. 先预测长度分布、按 τ=1 采样长度，再按长度生成 ----
    with torch.no_grad():
        # (1, 51) → 概率分布；每条候选肽**独立**采一个长度，
        # 等价于从联合分布 p(肽, k | 蛋白) 里采样
        length_probs = torch.softmax(length_predictor(cond), dim=-1)
        cond = cond.repeat(args.num_samples, 1)                     # (num_samples, D)
        length_probs = length_probs.repeat(args.num_samples, 1)     # (num_samples, 51)
        lengths = torch.multinomial(length_probs, 1).squeeze(1)     # (num_samples,) 各自一个 k
        samples = p_sample_loop(model, cond, PEPTIDE_LENGTH, lengths, device=DEVICE)

    # ---- 4. 打印：Sample 1 (采样长度 k): XXXX... ----
    # 长度是采样出来的，所以同一蛋白的多条候选肽长度通常不同
    for i in range(args.num_samples):
        k = int(lengths[i].item())
        peptide = decode_peptide(samples[i], max_len=k)
        print(f"Sample {i + 1} (采样长度 k={k}): {peptide}")


if __name__ == "__main__":
    main()
