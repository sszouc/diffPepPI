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
from data_utils import decode_peptide
from diffusion import p_sample_loop
from model import ConditionalDecoder, LengthPredictor

def load_embedding_from_pkl(seq: str, pkl_path: str) -> np.ndarray:
    with open(pkl_path, "rb") as f:
        embed_dict = pickle.load(f)
    if seq not in embed_dict:
        raise KeyError("Protein sequence not found in embedding file.")
    return embed_dict[seq]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
                        type=str,
                        default="checkpoints/model_epoch_100.pt")
    parser.add_argument(
        "--protein_seq",
                        type=str,
                        default="MADFEDRVSDEEKVRIAAKFITHAPPGEFNEVFNDVRLLLNNDNLLREGAAHAFAQYNMDQFTPVKIEGYDDQVLITEHGDLGNGRFLDPRNKISFKFDHLRKEASDPQPEDTESALKQWRDACDSALRAYVKDHYPNGFCTVYGKSIDGQQTIIACIESHQFQPKNFWNGRWRSEWKFTITPPTAQVAAVLKIQVHYYEDGNVQLVSHKDIQDSVQVSSDVQTAKEFIKIIENAENEYQTAISENYQTMSDTTFKALRRQLPVTRTKIDWNKILSYKIGKEMQNA")
    parser.add_argument(
        "--num_samples",
                        type=int,
                        default=5)
    args = parser.parse_args()

    embedding = load_embedding_from_pkl(args.protein_seq, EMBEDDING_PATH)
    if embedding.shape[-1] != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dim mismatch: expected {EMBEDDING_DIM}, got {embedding.shape[-1]}"
        )
    cond = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    model = ConditionalDecoder().to(DEVICE)
    length_predictor = LengthPredictor().to(DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt["decoder"])
    length_predictor.load_state_dict(ckpt["length_predictor"])
    model.eval()
    length_predictor.eval()

    with torch.no_grad():

        length_probs = torch.softmax(length_predictor(cond), dim=-1)
        cond = cond.repeat(args.num_samples, 1)
        length_probs = length_probs.repeat(args.num_samples, 1)
        lengths = torch.multinomial(length_probs, 1).squeeze(1)
        samples = p_sample_loop(model, cond, PEPTIDE_LENGTH, lengths, device=DEVICE)

    for i in range(args.num_samples):
        k = int(lengths[i].item())
        peptide = decode_peptide(samples[i], max_len=k)
        print(f"Sample {i + 1} (采样长度 k={k}): {peptide}")

if __name__ == "__main__":
    main()
