"""单条生成示例。

给一条蛋白序列，用训练好的权重生成若干条候选肽。
蛋白条件 = 该蛋白的 3D 结构 + 逐残基 ESM，经 StructureEncoder 编码成 memory。
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from config import DEVICE, PEPTIDE_LENGTH, STRUCT_MAX_LEN
from data_utils import (
    PropediaDataset, decode_peptide, load_records, make_collate, set_seed,
)
from diffusion import p_sample_loop
from model import ConditionalDecoder, LengthPredictor
from struct_encoder import StructureEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/model_best.pt")
    parser.add_argument("--protein_seq", default="ALQLTQSPSSLSASVGDRITITCRASQGVTSALAWYRQKPGSPPQLLIYDASSLESGVPSRFSGSGSGTEFTLTISTLRPEDFATYYCQQLHFYPHTFGGGTRVDVRRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYECEVTHQGLSSPVTKSFNRGEC",
                        help="蛋白序列；缺省时取验证侧第一个蛋白")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_records()
    seq = (args.protein_seq or "").strip().upper()
    if seq:
        matches = [r for r in records if r["PROTEIN_SEQ"] == seq]
        if not matches:
            raise SystemExit("该蛋白不在数据集里（无法提供结构/表征）。"
                             "请用数据集内的蛋白序列，或用 evaluate.py 的批量流程。")
        rec = matches[0]
    else:
        # 取长度接近中位的第一条，便于演示
        rec = sorted(records, key=lambda r: abs(len(r["PROTEIN_SEQ"]) - 237))[0]

    print(f"蛋白: {rec['id']}  (长度 {len(rec['PROTEIN_SEQ'])})")
    print(f"真实肽: {rec['PEPTIDE_SEQ']}")

    ds = PropediaDataset([0], [rec], load_structure=True)
    collate = make_collate(load_structure=True)
    batch = next(iter(DataLoader(ds, batch_size=1, collate_fn=collate)))
    batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
    n_valid = int(batch["node_mask"].sum().item())
    print(f"结构分支残基数: {n_valid}（上限 {STRUCT_MAX_LEN}）")

    decoder = ConditionalDecoder().to(DEVICE)
    length_predictor = LengthPredictor().to(DEVICE)
    struct_encoder = StructureEncoder().to(DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    if "struct_encoder" not in ckpt:
        raise SystemExit("这个 checkpoint 是旧结构（没有 struct_encoder），请先重新训练")
    decoder.load_state_dict(ckpt["decoder"])
    length_predictor.load_state_dict(ckpt["length_predictor"])
    struct_encoder.load_state_dict(ckpt["struct_encoder"])
    decoder.eval()
    length_predictor.eval()
    struct_encoder.eval()

    set_seed(args.seed)
    with torch.no_grad():
        s = struct_encoder(batch)
        lp = torch.softmax(length_predictor(batch["cond"]), dim=-1)
        memory = s["memory"].repeat(args.num_samples, 1, 1)
        mmask = s["memory_mask"].repeat(args.num_samples, 1)
        lp = lp.repeat(args.num_samples, 1)
        lengths = torch.multinomial(lp, 1).squeeze(1)
        samples = p_sample_loop(decoder, memory, PEPTIDE_LENGTH, lengths,
                                memory_mask=mmask, device=DEVICE)

    for i in range(args.num_samples):
        k = int(lengths[i].item())
        print(f"Sample {i+1} (k={k}): {decode_peptide(samples[i], max_len=k)}")


if __name__ == "__main__":
    main()
