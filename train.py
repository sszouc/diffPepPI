import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from config import (
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    T_SAMPLE_MIN,
    T_SAMPLE_MAX,
    PAD_IDX,
    X_IDX,
    MODEL_SAVE_DIR,
    PEPTIDE_LENGTH,
    SEED,
)
from data_utils import get_dataloaders, set_seed, decode_peptide
from diffusion import q_sample, p_sample_loop
from model import ConditionalDecoder, LengthPredictor

def train() -> None:
    set_seed(SEED)
    train_loader, val_loader = get_dataloaders(BATCH_SIZE)

    decoder = ConditionalDecoder().to(DEVICE)
    length_predictor = LengthPredictor().to(DEVICE)

    opt_dec = AdamW(decoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    opt_len = AdamW(length_predictor.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    best_ckpt = None
    for epoch in range(1, EPOCHS + 1):
        decoder.train()
        length_predictor.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}")

        for cond, x0, length in progress:
            cond = cond.to(DEVICE)
            x0 = x0.to(DEVICE)
            length = length.to(DEVICE)

            len_logits = length_predictor(cond)
            len_loss = F.cross_entropy(len_logits, length)
            opt_len.zero_grad(set_to_none=True)
            len_loss.backward()
            opt_len.step()

            t = torch.randint(
                T_SAMPLE_MIN,
                T_SAMPLE_MAX + 1,
                (x0.size(0),),
                device=DEVICE,
            )
            x_t = q_sample(x0, t)

            logits = decoder(x_t, t, cond)
            logits_flat = logits.view(-1, logits.size(-1))
            target_flat = x0.view(-1)

            target_flat = target_flat.masked_fill(target_flat == X_IDX, PAD_IDX)
            diff_loss = F.cross_entropy(
                logits_flat,
                target_flat,
                reduction="mean",
                ignore_index=PAD_IDX,
            )
            opt_dec.zero_grad(set_to_none=True)
            diff_loss.backward()
            opt_dec.step()

            with torch.no_grad():
                len_acc = (len_logits.argmax(dim=-1) == length).float().mean().item()
            progress.set_postfix(len=f"{len_loss.item():.3f}",
                                 acc=f"{len_acc:.2f}",
                                 diff=f"{diff_loss.item():.3f}")

        decoder.eval()
        length_predictor.eval()
        val_sum, val_tokens = 0.0, 0
        len_correct, len_total = 0, 0
        with torch.no_grad():
            for cond, x0, length in val_loader:
                cond = cond.to(DEVICE)
                x0 = x0.to(DEVICE)
                length = length.to(DEVICE)

                len_pred = length_predictor(cond).argmax(dim=-1)
                len_correct += int((len_pred == length).sum().item())
                len_total += int(length.numel())

                t = torch.randint(
                    T_SAMPLE_MIN,
                    T_SAMPLE_MAX + 1,
                    (x0.size(0),),
                    device=DEVICE,
                )
                x_t = q_sample(x0, t)

                target_flat = x0.view(-1).masked_fill(x0.view(-1) == X_IDX, PAD_IDX)
                valid = target_flat.ne(PAD_IDX)
                logits = decoder(x_t, t, cond)
                per = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), target_flat,
                    reduction="none", ignore_index=PAD_IDX,
                )
                val_sum += per[valid].sum().item()
                val_tokens += int(valid.sum().item())

        val_loss = val_sum / max(val_tokens, 1)
        len_acc = len_correct / max(len_total, 1)
        print(f"Validation loss: {val_loss:.4f} | 长度准确率: {len_acc:.4f} ({len_correct}/{len_total})")

        with torch.no_grad():
            for cond, _, length in val_loader:
                cond = cond.to(DEVICE)
                k_probs = torch.softmax(length_predictor(cond[:1]), dim=-1)
                k = torch.multinomial(k_probs, 1).squeeze(1)
                samples = p_sample_loop(decoder, cond[:1], PEPTIDE_LENGTH, k, device=DEVICE)
                peptide = decode_peptide(samples[0], max_len=int(k.item()))
                print(f"Sampled peptide (采样长度 {int(k.item())}, 真实长度 {int(length[0])}): {peptide}")
                break

        ckpt_path = os.path.join(MODEL_SAVE_DIR, f"model_epoch_{epoch}.pt")
        torch.save(
            {"decoder": decoder.state_dict(),
             "length_predictor": length_predictor.state_dict()},
            ckpt_path,
        )
        print(f"Saved checkpoint: {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt = ckpt_path
        print(
            f"Best validation so far: {best_val_loss:.4f} at {best_ckpt}"
            if best_ckpt
            else "Best validation so far: N/A"
        )

if __name__ == "__main__":
    train()
