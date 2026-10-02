#evaluate.py
"""批量生成肽并计算指标（含命令行入口）。

【当前生成逻辑：先预测长度分布，采样长度，再按该长度生成】
    LengthPredictor(cond) → 长度分布 p(k|蛋白) → **按 τ=1 的 softmax 采样**得到 k
    → p_sample_loop(..., k)：位置 0..k-1 被采样成残基、位置 k 放 <EOS>、其余为 <PAD>
    → decode_peptide(..., max_len=k) 输出恰好 k 个残基

    为什么采样而不是 argmax：argmax 永远取众数，会系统性偏向短肽
    （实测与真实长度分布的 TV 距离 0.28、KL 0.96），
    按 softmax 采样后 TV 降到 0.11、KL 降到 0.05；
    而且每条肽**独立**采一个 k，等价于从联合分布 p(肽, k | 蛋白) 里采样。

三项指标：
    1. KL(P_train || P_gen)：生成肽的 20 维氨基酸频率分布，与训练肽频率分布的 KL 散度；
       越小说明整体组成越接近参照集（参照集 = DATA_PATH 里的正样本肽）。
       注意它只比较**组成**，不代表序列合理性。
    2. Diversity：每个蛋白生成的若干条肽两两之间的平均 Levenshtein 距离，
       越大说明同一条件下的产出越多样（上限受生成长度限制）。
    3. Exact match：生成肽中有多少条与训练集里的某条肽**完全相同**（即“背题”比例），
       用 1 - 该比例近似“新颖性”。
       ⚠️ 该指标会被两件事污染：短肽天生容易与训练集撞上；以及 --split all 时
       评估蛋白里绝大多数本来就在训练集里。做“干净的新颖性”请用 --split val。

末尾还会打印“生成长度 vs 训练肽长度”的分布对比——本项目改成先预测长度后，
长度分布是判断长度预测器是否学到东西的关键指标。

运行示例：
    python evaluate.py                                     # 全部蛋白，结果缓存到 pkl
    python evaluate.py --split val                          # 只在“训练时没见过的蛋白”上评估
    python evaluate.py --limit 10 --num_samples 2           # 只跑前 10 个蛋白做快速自检
    python evaluate.py --dry_run                            # 只检查数据加载，不加载模型
"""
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


# 氨基酸字符 → 索引，用于统计频率
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def count_amino_acids(peptides: List[str]) -> np.ndarray:
    """统计一组肽里 20 种氨基酸各出现多少次。

    返回：(20,) int64 的计数向量，顺序与 AMINO_ACIDS 一致。
    非标准字符（例如 'X'）会被 AA_TO_IDX.get 返回 None 而**跳过**，
    既不计数也不报错。
    """
    counts = np.zeros(len(AMINO_ACIDS), dtype=np.int64)
    for pep in peptides:
        for aa in pep:
            idx = AA_TO_IDX.get(aa)
            if idx is not None:
                counts[idx] += 1
    return counts


def load_embeddings(pkl_path: str) -> Dict[str, np.ndarray]:
    """读取预计算的蛋白嵌入：pickle 字典 {蛋白序列: numpy(EMBEDDING_DIM,)}。"""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def compute_kl(p_real: np.ndarray, p_gen: np.ndarray, eps: float = 1e-8) -> float:
    """计算两个氨基酸频率分布之间的 KL 散度 KL(p_real || p_gen)。

    参数：
        p_real / p_gen : (20,) 的计数向量（不用事先归一化，函数内部会归一化）
        eps            : 平滑项，避免出现 0 导致 log(0) 或除零

    实现：
        · 先加 eps 再各自归一化，得到两个概率分布；
        · 返回 Σ p_real · log(p_real / p_gen)，单位是 nat。
    注意：这里比较的是**氨基酸组成分布**（20 维），不是序列级分布，
    所以它只能说明“组成像不像”，不能说明“序列合不合理”。
    """
    p_real = p_real.astype(np.float64) + eps
    p_gen = p_gen.astype(np.float64) + eps
    p_real = p_real / p_real.sum()
    p_gen = p_gen / p_gen.sum()
    return float(np.sum(p_real * np.log(p_real / p_gen)))


def levenshtein_distance(a: str, b: str) -> int:
    """两条字符串的编辑距离（插入 / 删除 / 替换 的代价都是 1）。

    经典动态规划：prev 存上一行，cur 是当前行。
    为省内存，先把较长的串放到 a（外层循环），较短的在 b（内层）。
    两条完全相同的肽返回 0，因此它同时可以用来判断“是否只是随机打乱”。
    """
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))          # 空串到 b 前 j 个字符的距离
    for i, ca in enumerate(a, start=1):
        cur = [i]                           # a 前 i 个字符到空串的距离
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1            # 插入
            delete = prev[j] + 1            # 删除
            sub = prev[j - 1] + (0 if ca == cb else 1)   # 相同则无需替换
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def average_pairwise_distance(peptides: List[str]) -> float:
    """一组肽的“两两平均 Levenshtein 距离”，作为多样性指标。

    仅在两两之间计算（i < j），是 O(n²) 的两两比较；
    本项目每个蛋白只生成 5 条，n=5，开销可忽略。
    样本数不足 2 时返回 0.0。
    """
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
    """统计生成肽中与训练集**完全一致**的条数，用来衡量“是否只是记住了训练集”。

    返回 (命中条数, 生成总条数)；1 - 命中/总数 就是报告里的“新颖性”。
    参照集由调用方给定：--split all 时是全部正样本肽；--split val 时只有训练肽。
    """
    train_set = set(train_peptides)         # 用集合把匹配从 O(n) 降到 O(1)
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
    """把 {蛋白序列: [生成的肽, ...]} 存成 pickle，避免每次重新采样（500 步很慢）。

    meta 记录本次生成用的设置（split / num_samples / limit / checkpoint），
    下次读缓存时用它判断"这份缓存还能不能代表当前请求"，避免换了设置却静默复用旧结果。
    """
    with open(cache_path, "wb") as f:
        pickle.dump({"meta": meta or {}, "peptides": peptides_by_protein}, f)


def load_cache(cache_path: str) -> Tuple[Dict[str, List[str]], Dict]:
    """读回生成结果缓存，返回 (peptides_by_protein, meta)。

    兼容早期版本写的"裸字典"格式（那时没有 meta，返回空 dict 表示来源未知）。
    """
    with open(cache_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "peptides" in obj and "meta" in obj:
        return obj["peptides"], obj["meta"]
    return obj, {}                      # 旧格式：裸字典，无 meta


def generate_in_batches(
    model: ConditionalDecoder,
    length_predictor: LengthPredictor,
    embeddings: Dict[str, np.ndarray],
    protein_seqs: List[str],
    num_samples: int,
    batch_size: int,
) -> Dict[str, List[str]]:
    """按蛋白分批生成肽，返回 {蛋白序列: [num_samples 条肽, ...]}。

    生成流程（先预测长度分布 → 采样长度 → 按长度生成）：
        1. 每个蛋白 → LengthPredictor 给出长度分布 p(k|蛋白)；
        2. 该分布复制 num_samples 份，**每条肽独立采一个长度 k**（τ=1 的 softmax 采样）——
           这等价于从联合分布 p(肽, k | 蛋白) 采样，也让同一蛋白的候选肽长度自然不同；
        3. p_sample_loop 按每条样本自己的 k 构造窗口（0..k-1 待采样，k 处 EOS，其余 PAD）；
        4. 解码时按各自的 max_len=k 截断 → 每条肽长度恰等于它自己的预测长度。

    参数：
        protein_seqs : 要生成条件的蛋白序列列表（调用方传入，通常是全部 5688 条，
                       或经 --limit / --split val 筛选后的若干条）
        num_samples  : 每个蛋白生成多少条
        batch_size   : 一次把多少个**蛋白**送进网络（显存不足就调小）

    关键一步是 repeat_interleave(num_samples, dim=0)：
        把一个蛋白的条件向量（和它的长度分布）各复制 num_samples 份，
        于是 (B_prot, D) → (B_prot*num, D)，一次采样就能同时得到"每个蛋白的多条候选肽"。
    """
    peptides_by_protein: Dict[str, List[str]] = {}
    model.eval()
    length_predictor.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(protein_seqs), batch_size), desc="Generating"):
            batch_seqs = protein_seqs[start : start + batch_size]
            batch_embs = []
            for seq in batch_seqs:
                emb = embeddings[seq]
                if emb.shape[-1] != EMBEDDING_DIM:      # 提前发现嵌入与配置不匹配（如 2560 vs 1280）
                    raise ValueError(
                        f"Embedding dim mismatch: expected {EMBEDDING_DIM}, got {emb.shape[-1]}"
                    )
                batch_embs.append(emb)

            # (B_prot, D) → float32 → 搬到设备
            cond = torch.tensor(np.stack(batch_embs), dtype=torch.float32).to(DEVICE)
            # ① 长度分布：(B_prot, 51)，softmax 归一化成概率（τ=1）
            length_probs = torch.softmax(length_predictor(cond), dim=-1)
            # ② 条件与长度分布一起复制 num_samples 份，再**逐条独立采样长度**
            cond = cond.repeat_interleave(num_samples, dim=0)                  # (B_prot*num, D)
            length_probs = length_probs.repeat_interleave(num_samples, dim=0)   # (B_prot*num, 51)
            lengths = torch.multinomial(length_probs, 1).squeeze(1)             # (B_prot*num,) 每条一个 k
            samples = p_sample_loop(model, cond, PEPTIDE_LENGTH, lengths, device=DEVICE)

            # 采样结果的顺序是 [蛋白0 的 n 条, 蛋白1 的 n 条, ...]，按下标切回每个蛋白
            for i, seq in enumerate(batch_seqs):
                start_idx = i * num_samples
                end_idx = start_idx + num_samples
                peps = [decode_peptide(samples[j], max_len=int(lengths[j].item()))
                        for j in range(start_idx, end_idx)]
                peptides_by_protein[seq] = peps

    return peptides_by_protein


def main() -> None:
    """命令行入口：解析参数 → 选择评估蛋白 → 生成肽 → 计算并打印指标。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model_epoch_100.pt",   # 与 config.EPOCHS=100 对应的最终权重
        help="Path to model checkpoint",            # 里面同时含 decoder 与 length_predictor
    )
    parser.add_argument("--num_samples", type=int, default=5)                 # 每个蛋白生成几条
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for protein embeddings",    # 指一次处理多少个蛋白条件
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="generated_peptides.pkl",
        help="Path to save/load generated peptides", # 生成结果缓存路径，传空字符串则不读写缓存
    )
    parser.add_argument(
        "--force_generate",
        action="store_true",
        help="Force regeneration even if cache exists",  # 忽略已有缓存，强制重新采样
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of proteins (0 means all)",   # 只跑前 N 个蛋白，方便自检
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "val"],
        help="all = 全部蛋白（含训练蛋白）；val = 只看训练时没见过的蛋白",  # 见下方说明
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only test data loading and training-peptide stats",   # 只查数据，不加载权重
    )
    args = parser.parse_args()

    embeddings = load_embeddings(EMBEDDING_PATH)
    protein_seqs = list(embeddings.keys())       # 嵌入表里的全部蛋白（5688 条）

    # ---- 选择评估蛋白 & 确定参照肽集合 ----
    # split=all：评估全部蛋白，参照集=全部正样本肽（与训练同一口径，便于和历史数字对比）
    # split=val：只评估“在训练集里一条肽都没出现过”的蛋白（真正没见过），
    #            并且参照集也换成**只用训练肽**——否则生成结果撞上验证肽会被误判成“背题”。
    #            这是判断新颖性/生成能力的干净口径。
    pairs_all = parse_fasta_pairs(DATA_PATH, only_positive=ONLY_POSITIVE)
    if args.split == "val":
        train_loader, _ = get_dataloaders(BATCH_SIZE)
        dataset_len = len(train_loader.dataset.dataset)
        if len(pairs_all) != dataset_len:
            raise RuntimeError(
                "FASTA 记录数与 PeptideDataset 样本数不一致（有蛋白缺少嵌入？），"
                "此时下标无法与 pairs 对齐，请先排查数据。"
            )
        # PeptideDataset 内部就是按 parse_fasta_pairs 的顺序建样本的，所以下标可以直接对齐
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

    if args.dry_run:                             # 自检路径：只打印数据规模就返回
        print(f"Loaded embeddings: {len(embeddings)}")
        print(f"Loaded train peptides: {len(train_peptides)}")
        print(f"Train AA counts: {train_counts.tolist()}")
        return

    # 缓存只按“蛋白 → 生成的肽”存储，所以这里既检查覆盖范围，也检查生成设置是否一致，
    # 任何一项不满足就重新生成，避免指标串味（用的还是旧 checkpoint / 旧 split 的结果）。
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
        # 权重文件里同时存了去噪网络与长度预测器（train.py 存的就是这个字典）
        # weights_only=True：只允许加载张量，避免反序列化任意对象（torch>=2.4 的安全默认）
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

    # ---- 汇总统计：氨基酸总频次、按蛋白平均的多样性 ----
    gen_counts = np.zeros(len(AMINO_ACIDS), dtype=np.int64)
    diversity_sum = 0.0
    diversity_groups = 0
    for seq in protein_seqs:
        peps = peptides_by_protein.get(seq, [])
        if not peps:
            continue
        gen_counts += count_amino_acids(peps)             # 累加所有生成肽的氨基酸计数
        diversity_sum += average_pairwise_distance(peps)  # 每个蛋白内部的平均两两距离
        diversity_groups += 1

    kl = compute_kl(train_counts, gen_counts)
    diversity = diversity_sum / max(diversity_groups, 1)  # 对所有蛋白取平均
    match_count, total_generated = compute_exact_match_stats(
        peptides_by_protein, train_peptides
    )
    match_rate = match_count / max(total_generated, 1)
    print(f"KL(P_train || P_gen) = {kl:.6f}")
    print(f"Diversity (avg pairwise Levenshtein) = {diversity:.4f}")
    print(f"Exact match to train: {match_count}/{total_generated} ({match_rate:.4%})")

    # ---- 长度报告：生成肽长度 vs 训练肽长度（本项目改成“先预测长度再生成”，
    #      所以长度分布是否贴近数据，是判断长度预测器有没有学会的关键指标）----
    gen_lengths = np.array([len(p) for peps in peptides_by_protein.values() for p in peps])
    ref_lengths = np.array([len(p) for p in train_peptides])
    if gen_lengths.size:
        print(f"生成长度:   mean {gen_lengths.mean():.2f} | median {np.median(gen_lengths):.1f} "
              f"| min {gen_lengths.min()} | max {gen_lengths.max()}")
    print(f"训练肽长度: mean {ref_lengths.mean():.2f} | median {np.median(ref_lengths):.1f} "
          f"| min {ref_lengths.min()} | max {ref_lengths.max()}")

    # 长度分布的两个距离指标（把长度当成 0..50 上的离散分布来比）：
    #   TV（全变差距离）= 0.5 * Σ|p_gen - p_ref|，越小越好（0 = 完全一致）
    #   KL(p_ref || p_gen) 同前面的氨基酸 KL，只是空间换成"长度"
    if gen_lengths.size:
        kmax = PEPTIDE_LENGTH
        p_gen = np.bincount(gen_lengths, minlength=kmax).astype(np.float64) / gen_lengths.size
        p_ref = np.bincount(ref_lengths, minlength=kmax).astype(np.float64) / ref_lengths.size
        tv = 0.5 * np.abs(p_gen - p_ref).sum()
        eps = 1e-12
        kl_len = float(np.sum(p_ref * np.log((p_ref + eps) / (p_gen + eps))))
        print(f"长度分布 vs 训练肽: TV距离 = {tv:.4f} | KL(train长度 || 生成长度) = {kl_len:.4f}")

    # 同一蛋白内部长度是否有变化（采样长度的直接效果；argmax 时这里恒为 0）
    spreads = [max(map(len, peps)) - min(map(len, peps))
               for peps in peptides_by_protein.values() if peps]
    if spreads:
        print(f"同一蛋白多条候选肽的长度极差: 平均 {np.mean(spreads):.2f}（0 表示长度完全不变）")


if __name__ == "__main__":
    main()
