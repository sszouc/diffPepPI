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

AMINO_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_fasta_pairs(fasta_path: str, only_positive: bool = False) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    current_protein = None
    current_peptide = None
    keep_record = True

    with open(fasta_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if keep_record and current_protein is not None and current_peptide is not None:
                    pairs.append((current_protein, current_peptide))
                current_protein, current_peptide = None, None

                keep_record = (not only_positive) or line.startswith(">positive")
                continue
            if line.startswith("Protein:"):

                current_protein = line.split("Protein:", 1)[1].strip()
            elif line.startswith("Peptide:"):
                current_peptide = line.split("Peptide:", 1)[1].strip()

    if keep_record and current_protein is not None and current_peptide is not None:
        pairs.append((current_protein, current_peptide))

    return pairs

def encode_peptide(seq: str, length: int = PEPTIDE_LENGTH) -> Tuple[List[int], int]:
    seq = seq.strip().upper()
    true_length = min(len(seq), length - 1)
    indices = []
    for i in range(true_length):
        aa = seq[i]
        indices.append(AMINO_TO_IDX.get(aa, X_IDX))
    indices.append(EOS_IDX)
    if len(indices) < length:
        indices.extend([PAD_IDX] * (length - len(indices)))
    return indices, true_length

def decode_peptide(indices, max_len: int | None = None) -> str:
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

    def __init__(self, fasta_path: str, embedding_path: str) -> None:

        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(
                f"Embedding file not found: {embedding_path}. "
                "Please precompute ESM2 embeddings first."
            )

        pairs = parse_fasta_pairs(fasta_path, only_positive=ONLY_POSITIVE)

        with open(embedding_path, "rb") as f:
            embed_dict: Dict[str, np.ndarray] = pickle.load(f)

        self.samples: List[Tuple[np.ndarray, List[int], int]] = []
        missing = 0
        for protein_seq, peptide_seq in pairs:
            if protein_seq not in embed_dict:
                missing += 1
                continue
            embedding = embed_dict[protein_seq]
            indices, true_length = encode_peptide(peptide_seq)
            self.samples.append((embedding, indices, true_length))

        if missing > 0:

            print(f"Warning: {missing} proteins missing embeddings; skipped.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        embedding, indices, true_length = self.samples[idx]
        embedding_t = torch.tensor(embedding, dtype=torch.float32)
        indices_t = torch.tensor(indices, dtype=torch.long)
        length_t = torch.tensor(true_length, dtype=torch.long)
        return embedding_t, indices_t, length_t

def get_dataloaders(
    batch_size: int,
    split_ratio: float = 0.9,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
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
