"""数据管线（Propedia 26）。

支持：
  - 残基级 ESM 表征（逐蛋白 .npy，懒加载）
  - 蛋白 3D 坐标（按 PROTEIN_SEQ 位置对齐，含 X 缺口的线性插值标记）
  - 界面残基标注（辅助监督用，**不能**当输入）
  - 用 seq100_clusters + PROTEIN_SEQ 做【肽级 + 蛋白级】无泄漏划分

依赖以下由预处理脚本产出的文件：
  data/propedia26_clean.csv          复合物标注
  data/residue_emb/{hash}.npy        逐蛋白残基级表征 (L, D)
  data/struct_cache/{id}.npz         ca / present / imputed / iface
  data/manifest.pkl                  对齐质量 + seq100_clusters

预处理：
  python split_pkl.py            # 7.2GB pkl -> 逐蛋白 npy
  python build_align.py          # PDB -> 位置坐标系对齐缓存
"""
from __future__ import annotations

import hashlib
import os
import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    AMINO_ACIDS,
    PAD_IDX,
    PEPTIDE_LENGTH,
    EOS_IDX,
    X_IDX,
    SEED,
    DATA_CSV,
    RESIDUE_EMB_DIR,
    STRUCT_CACHE_DIR,
    MANIFEST_PATH,
    SPLIT_RATIO,
    EMBEDDING_DIM,
    UNKNOWN_PROTEIN_TOKEN,
    STRUCT_MAX_LEN,
    USE_GLOBAL_TOKEN,
)

AMINO_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


# ---------------------------------------------------------------- 基础工具

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seq_hash(seq: str) -> str:
    """和 split_pkl.py 保持一致的哈希（文件名用）"""
    return hashlib.sha1(seq.strip().upper().encode()).hexdigest()[:16]


def encode_peptide(seq: str, length: int = PEPTIDE_LENGTH) -> Tuple[List[int], int]:
    """肽 -> 定长索引。第 L 位放 <EOS>，其余补 <PAD>。未知字符记为 <X>。"""
    seq = seq.strip().upper()
    true_length = min(len(seq), length - 1)
    indices = [AMINO_TO_IDX.get(seq[i], X_IDX) for i in range(true_length)]
    indices.append(EOS_IDX)
    if len(indices) < length:
        indices.extend([PAD_IDX] * (length - len(indices)))
    return indices, true_length


def decode_peptide(indices, max_len: Optional[int] = None) -> str:
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


# ---------------------------------------------------------------- 划分

def build_group_split(records: List[dict], ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    """肽级 + 蛋白级双约束的无泄漏划分。

    两个条目只要满足任一条件就不能分到不同集合：
      - PROTEIN_SEQ 相同（同一蛋白重复出现 -> 蛋白级泄漏）
      - seq100_clusters 相同（同一条肽重复出现 -> 肽级泄漏）

    做法：把条目看成图的节点，上述两个关系看成边，取连通分量，
    以「分量」为单位整体分配到 train/val。
    """
    n = len(records)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    first_by_protein: Dict[str, int] = {}
    first_by_cluster: Dict[str, int] = {}
    for i, r in enumerate(records):
        p = r["PROTEIN_SEQ"]
        if p in first_by_protein:
            union(i, first_by_protein[p])
        else:
            first_by_protein[p] = i
        c = r.get("seq100_clusters")
        if c is not None and not (isinstance(c, float) and np.isnan(c)):
            c = str(c)
            if c in first_by_cluster:
                union(i, first_by_cluster[c])
            else:
                first_by_cluster[c] = i

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)
    target = int(n * ratio)
    train, val = [], []
    for k in keys:
        idx = groups[k]
        if len(train) + len(idx) <= target:
            train.extend(idx)
        else:
            val.extend(idx)
    if not val and len(train) > 1:              # 退化保护
        val.append(train.pop())
    return sorted(train), sorted(val)


# ---------------------------------------------------------------- Dataset

class PropediaDataset(Dataset):
    """每个样本返回一个 dict：

    cond     (1280,)      全局条件：残基级表征的均值池化（长度分支的输入）
    x0       (51,)        肽的定长索引
    length   ()           肽真实长度
    prot_seq str          蛋白序列（供结构分支按需解码）
    pep_seq  str          肽序列
    id       str          复合物 id
    ca       (N,3)        按 PROTEIN_SEQ 位置排好的 CA 坐标（NaN 表示缺失）
    present  (N,)         有真实 CA
    imputed  (N,)         坐标是线性插值出来的
    iface    (N,)         是否界面残基（辅助监督标签，不要当输入）
    """

    def __init__(
        self,
        indices: Iterable[int],
        records: List[dict],
        emb_dir: str = RESIDUE_EMB_DIR,
        struct_dir: str = STRUCT_CACHE_DIR,
        load_structure: bool = True,
    ) -> None:
        self.indices = list(indices)
        self.records = records
        self.emb_dir = emb_dir
        self.struct_dir = struct_dir
        self.load_structure = load_structure
        self._missing_emb = 0

    def __len__(self) -> int:
        return len(self.indices)

    def _load_emb(self, prot_seq: str) -> Optional[np.ndarray]:
        """返回 (L, D) 的残基级表征；文件缺失时返回 None。"""
        path = os.path.join(self.emb_dir, f"{seq_hash(prot_seq)}.npy")
        if not os.path.exists(path):
            self._missing_emb += 1
            return None
        a = np.load(path)
        if a.ndim != 2 or a.shape[0] == 0:
            self._missing_emb += 1
            return None
        return np.asarray(a, dtype=np.float32)

    def __getitem__(self, i: int) -> dict:
        r = self.records[self.indices[i]]
        prot_seq, pep_seq = r["PROTEIN_SEQ"], r["PEPTIDE_SEQ"]

        emb = self._load_emb(prot_seq)
        cond = emb.mean(axis=0) if emb is not None else np.zeros(EMBEDDING_DIM, dtype=np.float32)

        x0, length = encode_peptide(pep_seq)
        out = {
            "cond": torch.from_numpy(np.asarray(cond, dtype=np.float32)),
            "x0": torch.tensor(x0, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "prot_seq": prot_seq,
            "pep_seq": pep_seq,
            "id": r["id"],
        }

        # 逐残基表征 + 氨基酸索引（结构编码器的节点特征）
        if emb is not None:
            out["esm"] = torch.from_numpy(emb.astype(np.float16, copy=False))
        else:
            out["esm"] = torch.zeros((len(prot_seq), EMBEDDING_DIM), dtype=torch.float16)
        out["aa_idx"] = torch.tensor(
            [AMINO_TO_IDX.get(c, len(AMINO_ACIDS)) for c in prot_seq], dtype=torch.long)

        if self.load_structure:
            npz = os.path.join(self.struct_dir, f"{r['id']}.npz")
            if os.path.exists(npz):
                d = np.load(npz)
                ca = d["ca"]
                present = d["present"]
                imputed = d["imputed"]
                iface = d["iface"]
            else:
                N = len(prot_seq)
                ca = np.full((N, 3), np.nan, dtype=np.float32)
                present = np.zeros(N, dtype=bool)
                imputed = np.zeros(N, dtype=bool)
                iface = np.zeros(N, dtype=bool)
            # NaN 坐标替换成 0，由 node_mask 屏蔽
            ca = np.nan_to_num(ca, nan=0.0)
            out["ca"] = torch.from_numpy(np.asarray(ca, dtype=np.float32))
            out["present"] = torch.from_numpy(np.asarray(present, dtype=bool))
            out["imputed"] = torch.from_numpy(np.asarray(imputed, dtype=bool))
            out["iface"] = torch.from_numpy(np.asarray(iface, dtype=bool))
        return out


def make_collate(load_structure: bool = True, max_len: int = STRUCT_MAX_LEN,
                 use_global_token: bool = USE_GLOBAL_TOKEN):
    """把 list[dict] 拼成 batch，结构部分 pad 到 batch 内最长（并截到 max_len）。"""

    def collate(items: List[dict]) -> dict:
        batch = {
            "cond": torch.stack([it["cond"] for it in items]),
            "x0": torch.stack([it["x0"] for it in items]),
            "length": torch.stack([it["length"] for it in items]),
            "prot_seq": [it["prot_seq"] for it in items],
            "pep_seq": [it["pep_seq"] for it in items],
            "id": [it["id"] for it in items],
        }
        if not load_structure:
            return batch

        Nmax = min(max(len(it["aa_idx"]) for it in items), max_len)
        B = len(items)
        esm = torch.zeros(B, Nmax, EMBEDDING_DIM, dtype=torch.float16)
        aa_idx = torch.zeros(B, Nmax, dtype=torch.long)
        ca = torch.zeros(B, Nmax, 3, dtype=torch.float32)
        present = torch.zeros(B, Nmax, dtype=torch.bool)
        imputed = torch.zeros(B, Nmax, dtype=torch.bool)
        iface = torch.zeros(B, Nmax, dtype=torch.bool)
        node_mask = torch.zeros(B, Nmax, dtype=torch.bool)

        for b, it in enumerate(items):
            n = min(len(it["aa_idx"]), Nmax)
            esm[b, :n] = it["esm"][:n]
            aa_idx[b, :n] = it["aa_idx"][:n]
            ca[b, :n] = it["ca"][:n]
            present[b, :n] = it["present"][:n]
            imputed[b, :n] = it["imputed"][:n]
            iface[b, :n] = it["iface"][:n]
            node_mask[b, :n] = it["present"][:n] | it["imputed"][:n]

        batch.update({
            "esm": esm, "aa_idx": aa_idx, "ca": ca,
            "present": present, "imputed": imputed, "iface": iface,
            "node_mask": node_mask, "use_global_token": use_global_token,
        })
        return batch

    return collate


# ---------------------------------------------------------------- 入口

def check_emb_shape(prot_seq: str, emb_dir: str = RESIDUE_EMB_DIR) -> bool:
    """残基级表征的行数是否等于序列长度。

    ESM2-650M 单次前向最多约 1022 个残基；本项目的抽取脚本实际上限为 1000，
    因此长度 > 1000 的蛋白其表征会被截断（行数 < 序列长度），与坐标/界面标注
    的索引不再一致。这类蛋白默认被过滤掉。
    """
    path = os.path.join(emb_dir, f"{seq_hash(prot_seq)}.npy")
    if not os.path.exists(path):
        return False
    try:
        a = np.load(path, mmap_mode="r")
    except Exception:
        return False
    return a.ndim == 2 and a.shape[0] == len(prot_seq)


def load_records(
    csv_path: str = DATA_CSV,
    manifest_path: str = MANIFEST_PATH,
    drop_bad_emb: bool = True,
) -> List[dict]:
    """读标注表，并合并 manifest 里的 seq100_clusters / 对齐质量。"""
    import pandas as pd

    df = pd.read_csv(csv_path, sep=";", low_memory=False)
    df.columns = df.columns.str.strip()
    if os.path.exists(manifest_path):
        man = pd.read_pickle(manifest_path)
        keep = man[["id", "seq100_clusters", "n_res", "n_ca", "n_iface", "status"]]
        df = df.merge(keep, on="id", how="left")
        df = df[df["status"].isin(["ok", "cached"])]
    else:
        df["seq100_clusters"] = None
        df["n_res"] = df["PROTEIN_SEQ"].str.len()
        df["n_ca"] = df["n_res"]
        df["n_iface"] = None
        df["status"] = "unknown"

    records = df.to_dict("records")
    for r in records:
        r["PROTEIN_SEQ"] = str(r["PROTEIN_SEQ"]).strip().upper()
        r["PEPTIDE_SEQ"] = str(r["PEPTIDE_SEQ"]).strip().upper()

    if drop_bad_emb and os.path.isdir(RESIDUE_EMB_DIR):
        seqs = list(dict.fromkeys(r["PROTEIN_SEQ"] for r in records))
        bad = {s for s in seqs if not check_emb_shape(s)}
        if bad:
            n_before = len(records)
            records = [r for r in records if r["PROTEIN_SEQ"] not in bad]
            print(f"[data] 过滤 {len(bad)} 个蛋白的 {n_before - len(records)} 条记录："
                  f"残基级表征被截断（长度超过 ESM 单次上限）")

    print(f"[data] 载入 {len(records)} 条复合物，"
          f"{len({r['PROTEIN_SEQ'] for r in records})} 个唯一蛋白")
    return records


def get_dataloaders(
    batch_size: int,
    split_ratio: float = SPLIT_RATIO,
    num_workers: int = 0,
    load_structure: bool = True,
    seed: int = SEED,
) -> Tuple[DataLoader, DataLoader]:
    """返回 (train_loader, val_loader)。划分已保证无肽级/蛋白级泄漏。"""
    set_seed(seed)
    records = load_records()
    tr, va = build_group_split(records, split_ratio, seed)

    train_set = PropediaDataset(tr, records, load_structure=load_structure)
    val_set = PropediaDataset(va, records, load_structure=load_structure)
    collate = make_collate(load_structure)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate)
    print(f"[data] 划分: train {len(tr)} / val {len(va)} "
          f"(按连通分量，肽级+蛋白级无泄漏)")
    return train_loader, val_loader
