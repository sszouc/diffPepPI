"""批量生成与指标。

条件链：蛋白结构 + 逐残基 ESM --StructureEncoder--> memory --交叉注意力--> 去噪网络
结构编码每个 batch 只做一次，采样的 100 步复用。
"""
import argparse
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    DEVICE, AMINO_ACIDS, PEPTIDE_LENGTH, BATCH_SIZE, SPLIT_RATIO, SEED,
)
from data_utils import (
    PropediaDataset, build_group_split, load_records, make_collate,
    decode_peptide, set_seed,
)
from diffusion import p_sample_loop
from model import ConditionalDecoder, LengthPredictor
from struct_encoder import StructureEncoder

AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def count_amino_acids(peptides: List[str]) -> np.ndarray:
    counts = np.zeros(len(AMINO_ACIDS), dtype=np.int64)
    for pep in peptides:
        for aa in pep:
            idx = AA_TO_IDX.get(aa)
            if idx is not None:
                counts[idx] += 1
    return counts


def compute_kl(p_real: np.ndarray, p_gen: np.ndarray, eps: float = 1e-8) -> float:
    p_real = p_real.astype(np.float64) + eps
    p_gen = p_gen.astype(np.float64) + eps
    p_real = p_real / p_real.sum()
    p_gen = p_gen / p_gen.sum()
    return float(np.sum(p_real * np.log(p_real / p_gen)))


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def average_pairwise_distance(peptides: List[str]) -> float:
    n = len(peptides)
    if n < 2:
        return 0.0
    total, count = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += levenshtein_distance(peptides[i], peptides[j])
            count += 1
    return total / count if count > 0 else 0.0


def compute_exact_match_stats(peptides_by_protein: Dict[str, List[str]],
                             train_peptides: List[str]) -> Tuple[int, int]:
    ref = set(train_peptides)
    match_count, total = 0, 0
    for peps in peptides_by_protein.values():
        total += len(peps)
        for pep in peps:
            if pep in ref:
                match_count += 1
    return match_count, total


def save_cache(cache_path: str, peptides_by_protein: Dict[str, List[str]],
               meta: Dict | None = None) -> None:
    with open(cache_path, "wb") as f:
        pickle.dump({"meta": meta or {}, "peptides": peptides_by_protein}, f)


def load_cache(cache_path: str) -> Tuple[Dict[str, List[str]], Dict]:
    with open(cache_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "peptides" in obj and "meta" in obj:
        return obj["peptides"], obj["meta"]
    return obj, {}


@torch.no_grad()
def generate_in_batches(decoder, length_predictor, struct_encoder,
                        records: List[dict], num_samples: int,
                        batch_size: int) -> Dict[str, List[str]]:
    """records: 每个待评估蛋白对应一条记录（取第一个复合物）。"""
    decoder.eval()
    length_predictor.eval()
    struct_encoder.eval()
    collate = make_collate(load_structure=True)
    ds = PropediaDataset(range(len(records)), records, load_structure=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    out: Dict[str, List[str]] = {}
    for batch in tqdm(loader, desc="Generating"):
        batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
        s = struct_encoder(batch)
        cond = batch["cond"]
        lp = torch.softmax(length_predictor(cond), dim=-1)

        memory = s["memory"].repeat_interleave(num_samples, dim=0)
        mmask = s["memory_mask"].repeat_interleave(num_samples, dim=0)
        lp = lp.repeat_interleave(num_samples, dim=0)
        lengths = torch.multinomial(lp, 1).squeeze(1)
        samples = p_sample_loop(decoder, memory, PEPTIDE_LENGTH, lengths,
                                memory_mask=mmask, device=DEVICE)

        for i, cid in enumerate(batch["id"]):
            lo, hi = i * num_samples, (i + 1) * num_samples
            out[batch["prot_seq"][i]] = [
                decode_peptide(samples[j], max_len=int(lengths[j].item()))
                for j in range(lo, hi)
            ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/model_best.pt")
    ap.add_argument("--num_samples", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--cache_path", default="generated_peptides.pkl")
    ap.add_argument("--force_generate", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default="val", choices=["all", "val"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    records = load_records()
    train_idx, val_idx = build_group_split(records, SPLIT_RATIO, SEED)

    def first_per_protein(indices):
        seen, out = set(), []
        for i in indices:
            s = records[i]["PROTEIN_SEQ"]
            if s not in seen:
                seen.add(s)
                out.append(records[i])
        return out

    if args.split == "val":
        eval_records = first_per_protein(val_idx)
        ref_peptides = [records[i]["PEPTIDE_SEQ"] for i in train_idx]
        print(f"[split=val] 未见过的蛋白 {len(eval_records)} 个 | "
              f"参照集=训练肽 {len(ref_peptides)} 条")
    else:
        eval_records = first_per_protein(range(len(records)))
        ref_peptides = [r["PEPTIDE_SEQ"] for r in records]
        print(f"[split=all] 蛋白 {len(eval_records)} 个 | 参照集=全部肽 {len(ref_peptides)} 条")

    if args.limit > 0:
        eval_records = eval_records[: args.limit]

    if args.dry_run:
        print(f"records={len(records)} train={len(train_idx)} val={len(val_idx)}")
        print(f"eval_proteins={len(eval_records)}")
        return

    peptides_by_protein = None
    meta_want = {"split": args.split, "num_samples": args.num_samples,
                 "limit": args.limit, "checkpoint": args.checkpoint}
    if not args.force_generate and args.cache_path and os.path.exists(args.cache_path):
        cached, meta = load_cache(args.cache_path)
        problems = []
        missing = sum(1 for r in eval_records if r["PROTEIN_SEQ"] not in cached)
        if missing:
            problems.append(f"缺少 {missing} 个待评估蛋白")
        for key in ("split", "num_samples", "checkpoint"):
            if meta and meta.get(key) != meta_want[key]:
                problems.append(f"{key} 不一致")
        if problems:
            print("[提示] 缓存不可用：" + "；".join(problems) + " → 重新生成")
        else:
            peptides_by_protein = cached
            print(f"命中缓存：{args.cache_path}")

    if peptides_by_protein is None:
        decoder = ConditionalDecoder().to(DEVICE)
        length_predictor = LengthPredictor().to(DEVICE)
        struct_encoder = StructureEncoder().to(DEVICE)
        ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
        decoder.load_state_dict(ckpt["decoder"])
        length_predictor.load_state_dict(ckpt["length_predictor"])
        if "struct_encoder" not in ckpt:
            raise SystemExit("这个 checkpoint 是旧结构（没有 struct_encoder），请先重新训练")
        struct_encoder.load_state_dict(ckpt["struct_encoder"])
        # 种子必须在数据加载之后设，否则会被 data_utils 的 set_seed 覆盖
        set_seed(args.seed)
        print(f"[seed={args.seed}] 采样使用该随机种子")
        peptides_by_protein = generate_in_batches(
            decoder, length_predictor, struct_encoder, eval_records,
            args.num_samples, args.batch_size)
        if args.cache_path:
            save_cache(args.cache_path, peptides_by_protein, meta=meta_want)

    gen_counts = np.zeros(len(AMINO_ACIDS), dtype=np.int64)
    diversity_sum, diversity_groups = 0.0, 0
    for r in eval_records:
        peps = peptides_by_protein.get(r["PROTEIN_SEQ"], [])
        if not peps:
            continue
        gen_counts += count_amino_acids(peps)
        diversity_sum += average_pairwise_distance(peps)
        diversity_groups += 1

    kl = compute_kl(count_amino_acids(ref_peptides), gen_counts)
    diversity = diversity_sum / max(diversity_groups, 1)
    match_count, total_gen = compute_exact_match_stats(peptides_by_protein, ref_peptides)
    print(f"KL(P_train || P_gen) = {kl:.6f}")
    print(f"Diversity (avg pairwise Levenshtein) = {diversity:.4f}")
    print(f"Exact match to train: {match_count}/{total_gen} ({match_count/max(total_gen,1):.4%})")

    gen_lengths = np.array([len(p) for peps in peptides_by_protein.values() for p in peps])
    ref_lengths = np.array([len(p) for p in ref_peptides])
    if gen_lengths.size:
        print(f"生成长度:   mean {gen_lengths.mean():.2f} | median {np.median(gen_lengths):.1f} "
              f"| min {gen_lengths.min()} | max {gen_lengths.max()}")
    print(f"训练肽长度: mean {ref_lengths.mean():.2f} | median {np.median(ref_lengths):.1f} "
          f"| min {ref_lengths.min()} | max {ref_lengths.max()}")

    if gen_lengths.size:
        kmax = PEPTIDE_LENGTH
        p_gen = np.bincount(gen_lengths, minlength=kmax).astype(np.float64) / gen_lengths.size
        p_ref = np.bincount(ref_lengths, minlength=kmax).astype(np.float64) / ref_lengths.size
        tv = 0.5 * np.abs(p_gen - p_ref).sum()
        eps = 1e-12
        kl_len = float(np.sum(p_ref * np.log((p_ref + eps) / (p_gen + eps))))
        print(f"长度分布 vs 训练肽: TV距离 = {tv:.4f} | KL(train长度 || 生成长度) = {kl_len:.4f}")

    spreads = [max(map(len, p)) - min(map(len, p))
               for p in peptides_by_protein.values() if p]
    if spreads:
        print(f"同一蛋白多条候选肽的长度极差: 平均 {np.mean(spreads):.2f}")


if __name__ == "__main__":
    main()
