import argparse
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from config import (
    DEVICE,
    AMINO_ACIDS,
    PEPTIDE_LENGTH,
    EMBEDDING_PATH,
    DATA_PATH,
    EMBEDDING_DIM,
    ONLY_POSITIVE,
    BATCH_SIZE,
)
from data_utils import parse_fasta_pairs, decode_peptide, get_dataloaders
from diffusion import p_sample_loop
from model import ConditionalDecoder, LengthPredictor

AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def count_amino_acids(peptides: List[str]) -> np.ndarray:
    counts = np.zeros(len(AMINO_ACIDS), dtype=np.int64)
    for pep in peptides:
        for aa in pep:
            idx = AA_TO_IDX.get(aa)
            if idx is not None:
                counts[idx] += 1
    return counts

def load_embeddings(pkl_path: str) -> Dict[str, np.ndarray]:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

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
    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += levenshtein_distance(peptides[i], peptides[j])
            count += 1
    return total / count if count > 0 else 0.0

def compute_exact_match_stats(
    peptides_by_protein: Dict[str, List[str]],
    train_peptides: List[str],
) -> Tuple[int, int]:
    train_set = set(train_peptides)
    match_count = 0
    total = 0
    for peps in peptides_by_protein.values():
        total += len(peps)
        for pep in peps:
            if pep in train_set:
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

def generate_in_batches(
    model: ConditionalDecoder,
    length_predictor: LengthPredictor,
    embeddings: Dict[str, np.ndarray],
    protein_seqs: List[str],
    num_samples: int,
    batch_size: int,
) -> Dict[str, List[str]]:
    peptides_by_protein: Dict[str, List[str]] = {}
    model.eval()
    length_predictor.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(protein_seqs), batch_size), desc="Generating"):
            batch_seqs = protein_seqs[start : start + batch_size]
            batch_embs = []
            for seq in batch_seqs:
                emb = embeddings[seq]
                if emb.shape[-1] != EMBEDDING_DIM:
                    raise ValueError(
                        f"Embedding dim mismatch: expected {EMBEDDING_DIM}, got {emb.shape[-1]}"
                    )
                batch_embs.append(emb)

            cond = torch.tensor(np.stack(batch_embs), dtype=torch.float32).to(DEVICE)

            length_probs = torch.softmax(length_predictor(cond), dim=-1)

            cond = cond.repeat_interleave(num_samples, dim=0)
            length_probs = length_probs.repeat_interleave(num_samples, dim=0)
            lengths = torch.multinomial(length_probs, 1).squeeze(1)
            samples = p_sample_loop(model, cond, PEPTIDE_LENGTH, lengths, device=DEVICE)

            for i, seq in enumerate(batch_seqs):
                start_idx = i * num_samples
                end_idx = start_idx + num_samples
                peps = [decode_peptide(samples[j], max_len=int(lengths[j].item()))
                        for j in range(start_idx, end_idx)]
                peptides_by_protein[seq] = peps

    return peptides_by_protein

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model_epoch_100.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for protein embeddings",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="generated_peptides.pkl",
        help="Path to save/load generated peptides",
    )
    parser.add_argument(
        "--force_generate",
        action="store_true",
        help="Force regeneration even if cache exists",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of proteins (0 means all)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "val"],
        help="all = 全部蛋白（含训练蛋白）；val = 只看训练时没见过的蛋白",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only test data loading and training-peptide stats",
    )
    args = parser.parse_args()

    embeddings = load_embeddings(EMBEDDING_PATH)
    protein_seqs = list(embeddings.keys())

    pairs_all = parse_fasta_pairs(DATA_PATH, only_positive=ONLY_POSITIVE)
    if args.split == "val":
        train_loader, _ = get_dataloaders(BATCH_SIZE)
        dataset_len = len(train_loader.dataset.dataset)
        if len(pairs_all) != dataset_len:
            raise RuntimeError(
                "FASTA 记录数与 PeptideDataset 样本数不一致（有蛋白缺少嵌入？），"
                "此时下标无法与 pairs 对齐，请先排查数据。"
            )

        train_idx = sorted(train_loader.dataset.indices)
        trained_proteins = {pairs_all[i][0] for i in train_idx}
        protein_seqs = [s for s in protein_seqs if s not in trained_proteins]
        train_peptides = [pairs_all[i][1] for i in train_idx]
        print(f"[split=val] 训练没见过的蛋白 {len(protein_seqs)} 个 | 参照集=训练肽 {len(train_peptides)} 条")
    else:
        train_peptides = [pep for _, pep in pairs_all]
        print(f"[split=all] 评估蛋白 {len(protein_seqs)} 个（含训练蛋白）| 参照集=正样本肽 {len(train_peptides)} 条")

    if args.limit > 0:
        protein_seqs = protein_seqs[: args.limit]

    train_counts = count_amino_acids(train_peptides)

    if args.dry_run:
        print(f"Loaded embeddings: {len(embeddings)}")
        print(f"Loaded train peptides: {len(train_peptides)}")
        print(f"Train AA counts: {train_counts.tolist()}")
        return

    peptides_by_protein = None
    meta_want = {
        "split": args.split,
        "num_samples": args.num_samples,
        "limit": args.limit,
        "checkpoint": args.checkpoint,
    }
    if not args.force_generate and args.cache_path and os.path.exists(args.cache_path):
        cached, meta = load_cache(args.cache_path)
        problems = []
        missing = sum(1 for s in protein_seqs if s not in cached)
        if missing:
            problems.append(f"缺少 {missing} 个待评估蛋白")
        for key in ("split", "num_samples", "checkpoint"):
            if meta and meta.get(key) != meta_want[key]:
                problems.append(f"{key} 不一致（缓存 {meta.get(key)} ≠ 本次 {meta_want[key]}）")
        if problems:
            print("[提示] 缓存不可用：" + "；".join(problems) + " → 重新生成")
        else:
            peptides_by_protein = cached
            print(f"命中缓存：{args.cache_path}（{len(cached)} 个蛋白）")

    if peptides_by_protein is None:
        model = ConditionalDecoder().to(DEVICE)
        length_predictor = LengthPredictor().to(DEVICE)

        ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["decoder"])
        length_predictor.load_state_dict(ckpt["length_predictor"])
        peptides_by_protein = generate_in_batches(
            model,
            length_predictor,
            embeddings,
            protein_seqs,
            args.num_samples,
            args.batch_size,
        )
        if args.cache_path:
            save_cache(args.cache_path, peptides_by_protein, meta=meta_want)

    gen_counts = np.zeros(len(AMINO_ACIDS), dtype=np.int64)
    diversity_sum = 0.0
    diversity_groups = 0
    for seq in protein_seqs:
        peps = peptides_by_protein.get(seq, [])
        if not peps:
            continue
        gen_counts += count_amino_acids(peps)
        diversity_sum += average_pairwise_distance(peps)
        diversity_groups += 1

    kl = compute_kl(train_counts, gen_counts)
    diversity = diversity_sum / max(diversity_groups, 1)
    match_count, total_generated = compute_exact_match_stats(
        peptides_by_protein, train_peptides
    )
    match_rate = match_count / max(total_generated, 1)
    print(f"KL(P_train || P_gen) = {kl:.6f}")
    print(f"Diversity (avg pairwise Levenshtein) = {diversity:.4f}")
    print(f"Exact match to train: {match_count}/{total_generated} ({match_rate:.4%})")

    gen_lengths = np.array([len(p) for peps in peptides_by_protein.values() for p in peps])
    ref_lengths = np.array([len(p) for p in train_peptides])
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

    spreads = [max(map(len, peps)) - min(map(len, peps))
               for peps in peptides_by_protein.values() if peps]
    if spreads:
        print(f"同一蛋白多条候选肽的长度极差: 平均 {np.mean(spreads):.2f}（0 表示长度完全不变）")

if __name__ == "__main__":
    main()
