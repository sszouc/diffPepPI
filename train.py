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
    TRAIN_T_STEPS,
    VAL_T_STEPS,
    EOS_IDX,
    OUT_CLASSES,
    MODEL_SAVE_DIR,
    PEPTIDE_LENGTH,
    EARLY_STOP_PATIENCE,
    EARLY_STOP_MIN_DELTA,
    SEED,
)
from data_utils import get_dataloaders, set_seed, decode_peptide
from diffusion import q_sample, p_sample_loop
from model import ConditionalDecoder, LengthPredictor

def masked_loss(logits: torch.Tensor, x0: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """只在<b>被加噪的位置</b>上算交叉熵（外加 <EOS> 位置）。

    logits : (B, 51, 21)
    x0     : (B, 51) 干净目标，取值 0~23
    mask   : (B, 51) 被替换成 <MASK> 的布尔位置

    参与位置 = 被加噪位 ∪ <EOS> 位
      - 被加噪位：扩散去噪的核心监督信号
      - <EOS> 位：永不加噪，但要学会"到此结束"；否则它在生成时完全无监督
      - <PAD>/<X> 位：既不加噪、目标也不合法，天然被排除
    """
    tgt = x0.view(-1)
    keep = mask.view(-1) | tgt.eq(EOS_IDX)
    per = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        tgt.clamp(0, OUT_CLASSES - 1),
        reduction="none",
    )
    return per[keep].sum() / keep.sum().clamp_min(1)

def train() -> None:
    set_seed(SEED)
    train_loader, val_loader = get_dataloaders(BATCH_SIZE)

    decoder = ConditionalDecoder().to(DEVICE)
    length_predictor = LengthPredictor().to(DEVICE)

    opt_dec = AdamW(decoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    opt_len = AdamW(length_predictor.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    best_epoch = 0
    best_path = None
    patience_left = EARLY_STOP_PATIENCE
    stop_epoch = EPOCHS

    for epoch in range(1, EPOCHS + 1):
        decoder.train()
        length_predictor.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}")

        for cond, x0, length in progress:
            #cond是蛋白质的1280维特征，x0是肽，length是真实长度
            #condition，条件生成
            # cond, x0, length 的第一维都是batch
            cond = cond.to(DEVICE)
            x0 = x0.to(DEVICE)
            length = length.to(DEVICE)

            len_logits = length_predictor(cond)
            len_loss = F.cross_entropy(len_logits, length)
            opt_len.zero_grad(set_to_none=True)
            len_loss.backward()
            opt_len.step()
            # 先预测长度

            t = torch.randint(TRAIN_T_STEPS, (x0.size(0),), device=DEVICE)
            #随机生成时间步t
            x_t, mask = q_sample(x0, t, return_mask=True)
            #得到xt，时间步t情况下的肽情况；mask 是被替换成 <MASK> 的位置
            #<PAD>/<X>/<EOS> 三类永不加噪，所以 mask 只会落在真实残基上

            logits = decoder(x_t, t, cond)
            #logits  (B,51,21)

            # 只对被加噪的位置算损失：
            diff_loss = masked_loss(logits, x0, mask)

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
        len_correct, len_total = 0, 0
        # 每个时间步单独累计，最后对各时间步取等权平均（而不是把所有时间步混在一起）
        step_sum = {tv: 0.0 for tv in VAL_T_STEPS}
        step_cnt = {tv: 0 for tv in VAL_T_STEPS}
        val_gen = torch.Generator(device=DEVICE).manual_seed(SEED)
        # 验证是纯前向，开 tf32 加速；训练保持 fp32 以保证梯度精度
        if DEVICE.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        with torch.no_grad():
            for cond, x0, length in val_loader:
                cond = cond.to(DEVICE)
                x0 = x0.to(DEVICE)
                length = length.to(DEVICE)

                len_pred = length_predictor(cond).argmax(dim=-1)
                len_correct += int((len_pred == length).sum().item())
                len_total += int(length.numel())

                # 验证时用与训练完全一致的口径（加噪位 + <EOS> 位）
                B = x0.size(0)

                for tv in VAL_T_STEPS:
                    t = torch.full((B,), tv, device=DEVICE)

                    # 整个 batch 一次前向（验证是 no_grad 前向，显存远小于训练）
                    rand = torch.rand(x0.shape, dtype=torch.float32,
                                      device=DEVICE, generator=val_gen)
                    x_t, mask = q_sample(x0, t, rand=rand, return_mask=True)
                    logits = decoder(x_t, t, cond)

                    keep = mask.view(-1) | x0.view(-1).eq(EOS_IDX)
                    n_k = int(keep.sum())
                    if n_k:
                        per = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            x0.view(-1).clamp(0, OUT_CLASSES - 1),
                            reduction="none",
                        )
                        step_sum[tv] += per[keep].sum().item()
                        step_cnt[tv] += n_k

        if DEVICE.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False

        step_losses = [step_sum[tv] / step_cnt[tv] for tv in VAL_T_STEPS if step_cnt[tv] > 0]
        val_loss = sum(step_losses) / max(len(step_losses), 1)
        len_acc = len_correct / max(len_total, 1)

        print(f"Val loss: {val_loss:.4f} | 长度准确率: {len_acc:.4f} ")

        with torch.no_grad():
            for cond, _, length in val_loader:
                cond = cond.to(DEVICE)
                k_probs = torch.softmax(length_predictor(cond[:1]), dim=-1)
                k = torch.multinomial(k_probs, 1).squeeze(1)
                samples = p_sample_loop(decoder, cond[:1], PEPTIDE_LENGTH, k, device=DEVICE)
                peptide = decode_peptide(samples[0], max_len=int(k.item()))
                print(f"Sampled peptide (采样长度 {int(k.item())}, 真实长度 {int(length[0])}): {peptide}")
                break

        last_path = os.path.join(MODEL_SAVE_DIR, "model_last.pt")
        torch.save(
            {"decoder": decoder.state_dict(),
             "length_predictor": length_predictor.state_dict()},
            last_path,
        )

        if val_loss < best_val_loss - EARLY_STOP_MIN_DELTA:
            best_val_loss = val_loss
            best_epoch = epoch
            best_path = os.path.join(MODEL_SAVE_DIR, "model_best.pt")
            torch.save(
                {"decoder": decoder.state_dict(),
                 "length_predictor": length_predictor.state_dict(),
                 "epoch": epoch,
                 "val_loss": val_loss,
                 "val_t_steps": VAL_T_STEPS,
                 "seed": SEED},
                best_path,
            )
            patience_left = EARLY_STOP_PATIENCE
            print(f"[改进] 验证损失 {val_loss:.4f} 为当前最优，已保存 {best_path}")
        else:
            patience_left -= 1
            print(f"验证损失未改进（最优 {best_val_loss:.4f} @ epoch {best_epoch}，"
                  f"还等 {patience_left}/{EARLY_STOP_PATIENCE} 轮）")
            if patience_left <= 0:
                stop_epoch = epoch
                print(f"连续 {EARLY_STOP_PATIENCE} 轮没有改进，提前停止训练。")
                break

        print(f"Saved checkpoint: {last_path}")

    if best_path is None:
        print("没有保存任何 checkpoints/model_best.pt，请检查验证损失是否一直为 NaN。")
    else:
        print(f"训练结束：epoch {stop_epoch} 停 | 最优 epoch {best_epoch}，验证损失 {best_val_loss:.4f} -> {best_path}")

if __name__ == "__main__":
    train()
