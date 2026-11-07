"""训练主循环。

三个模块：
  StructureEncoder    蛋白 3D 结构 + 逐残基 ESM -> 交叉注意力 memory + 界面 logits
  ConditionalDecoder  去噪网络（消费 memory）
  LengthPredictor     肽长分布

两个损失：
  diff_loss   只在【被加噪位置 + <EOS>】上的交叉熵
  iface_loss  逐残基界面预测的 BCE（辅助任务，用真界面标注做标签）

关键效率点：结构编码与时间步无关，所以**每个 batch 只编码一次**，
扩散的 100 步（以及验证的 4 个时间步）复用同一份 memory。
采样时也只在入口编码一次。
"""
import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from config import (
    DEVICE, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EPOCHS,
    TRAIN_T_STEPS, VAL_T_STEPS, EOS_IDX, OUT_CLASSES,
    MODEL_SAVE_DIR, PEPTIDE_LENGTH, EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA,
    IFACE_LOSS_WEIGHT, GRAD_CLIP, SEED,
)
from data_utils import get_dataloaders, set_seed, decode_peptide
from diffusion import q_sample, p_sample_loop
from model import ConditionalDecoder, LengthPredictor
from struct_encoder import StructureEncoder


def masked_loss(logits: torch.Tensor, x0: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """只在<b>被加噪的位置</b>上算交叉熵（外加 <EOS> 位置）。

    logits : (B, 51, 21)
    x0     : (B, 51) 干净目标，取值 0~23
    mask   : (B, 51) 被替换成 <MASK> 的布尔位置
    """
    tgt = x0.view(-1)
    keep = mask.view(-1) | tgt.eq(EOS_IDX)
    per = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        tgt.clamp(0, OUT_CLASSES - 1),
        reduction="none",
    )
    return per[keep].sum() / keep.sum().clamp_min(1)


def iface_loss(logits: torch.Tensor, batch: dict) -> torch.Tensor:
    """逐残基界面预测 BCE。只在【有真实坐标】的残基上算。"""
    target = batch["iface"].float()
    valid = batch["present"] & batch["node_mask"]
    if not valid.any():
        return logits.sum() * 0.0
    per = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return per[valid].sum() / valid.sum().clamp_min(1)


def move_batch(batch: dict) -> dict:
    """把 batch 里的张量送到 DEVICE（字符串/list 保持原样）。"""
    out = {}
    for k, v in batch.items():
        out[k] = v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v
    return out


def compute_length_prior(dataset, n_classes: int = PEPTIDE_LENGTH) -> torch.Tensor:
    """训练集的肽长经验分布（平滑后归一化）。

    用作长度分支的"边际基线"：一个完全忽略蛋白条件、只按训练集长度分布
    预测的模型能拿到多少 NLL。模型的 NLL 只有明显低于它，才说明长度确实
    用上了蛋白条件。
    """
    cnt = torch.zeros(n_classes, dtype=torch.float64)
    for i in dataset.indices:
        L = len(dataset.records[i]["PEPTIDE_SEQ"])
        if 0 <= L < n_classes:
            cnt[L] += 1
    p = (cnt + 1e-9) / (cnt.sum() + 1e-9 * n_classes)
    return p.float()


def train() -> None:
    set_seed(SEED)
    train_loader, val_loader = get_dataloaders(BATCH_SIZE)

    # 长度先验（边际基线），在整个训练期间固定
    len_prior = compute_length_prior(train_loader.dataset).to(DEVICE)
    log_prior = torch.log(len_prior.clamp_min(1e-12))
    marginal_nll = float(-(len_prior * log_prior).sum())     # 边际分布的熵
    print(f"[length] 训练集长度先验的熵（边际基线 NLL 的理论值）= {marginal_nll:.4f} nats")

    decoder = ConditionalDecoder().to(DEVICE)
    length_predictor = LengthPredictor().to(DEVICE)
    struct_encoder = StructureEncoder().to(DEVICE)

    # 去噪网络与结构编码器联合训练（同一次前向，梯度耦合）
    opt_main = AdamW(
        list(decoder.parameters()) + list(struct_encoder.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    opt_len = AdamW(length_predictor.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    best_epoch = 0
    best_path = None
    patience_left = EARLY_STOP_PATIENCE
    stop_epoch = EPOCHS

    n_bad = 0
    for epoch in range(1, EPOCHS + 1):
        bad_this_epoch = 0
        decoder.train()
        length_predictor.train()
        struct_encoder.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in progress:
            batch = move_batch(batch)

            # ---- 结构编码（每 batch 一次）
            s = struct_encoder(batch)
            memory, memory_mask = s["memory"], s["memory_mask"]

            # ---- 长度分支
            len_logits = length_predictor(batch["cond"])
            len_loss = F.cross_entropy(len_logits, batch["length"])
            opt_len.zero_grad(set_to_none=True)
            len_loss.backward()
            opt_len.step()

            # ---- 去噪分支
            x0 = batch["x0"]
            t = torch.randint(TRAIN_T_STEPS, (x0.size(0),), device=DEVICE)
            x_t, mask = q_sample(x0, t, return_mask=True)
            logits = decoder(x_t, t, memory, memory_mask)
            diff_loss = masked_loss(logits, x0, mask)

            # ---- 界面辅助损失
            aux = iface_loss(s["iface_logits"], batch)
            loss = diff_loss + IFACE_LOSS_WEIGHT * aux

            opt_main.zero_grad(set_to_none=True)
            loss.backward()
            # 梯度裁剪 + 非有限值保护：单个异常 batch 不应该污染全部参数
            gnorm = torch.nn.utils.clip_grad_norm_(
                list(decoder.parameters()) + list(struct_encoder.parameters()),
                max_norm=GRAD_CLIP,
            )
            if not torch.isfinite(loss) or not torch.isfinite(gnorm):
                opt_main.zero_grad(set_to_none=True)
                n_bad += 1
                bad_this_epoch += 1
                continue
            opt_main.step()

            with torch.no_grad():
                # 不再报 argmax 准确率（生成时是采样，argmax 命中率既与生成方式矛盾，
                # 又和"永远猜众数"的平凡基线同量级）。这里只看 top-5，作监控用。
                top5 = (len_logits.topk(5, dim=-1).indices
                        == batch["length"][:, None]).any(-1).float().mean().item()
            progress.set_postfix(
                len=f"{len_loss.item():.3f}",
                top5=f"{top5:.2f}",
                diff=f"{diff_loss.item():.3f}",
                iface=f"{aux.item():.3f}",
            )

        # ------------------------------ 验证
        decoder.eval()
        length_predictor.eval()
        struct_encoder.eval()
        # 长度分支的指标：NLL / top-5 / 期望长度 MAE，并与"不看蛋白的边际基线"对比
        len_nll_sum, len_top5, len_mae_sum, len_marg_sum, len_n = 0.0, 0, 0.0, 0.0, 0
        step_sum = {tv: 0.0 for tv in VAL_T_STEPS}
        step_cnt = {tv: 0 for tv in VAL_T_STEPS}
        iface_sum, iface_cnt = 0.0, 0
        val_gen = torch.Generator(device=DEVICE).manual_seed(SEED)
        if DEVICE.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch(batch)
                # 整个 batch 的结构只编码一次，4 个时间步复用
                s = struct_encoder(batch)
                memory, memory_mask = s["memory"], s["memory_mask"]

                # ---- 长度分支指标
                lg = length_predictor(batch["cond"])
                tgt_len = batch["length"]
                n_b = tgt_len.numel()
                len_nll_sum += float(F.cross_entropy(lg, tgt_len, reduction="sum"))
                len_top5 += int((lg.topk(5, dim=-1).indices
                                 == tgt_len[:, None]).any(-1).sum())
                idx = torch.arange(lg.size(-1), device=DEVICE, dtype=lg.dtype)
                exp_len = (torch.softmax(lg, dim=-1) * idx).sum(-1)
                len_mae_sum += float((exp_len - tgt_len.float()).abs().sum())
                # 边际基线：忽略蛋白条件，直接用训练集的长度分布预测
                len_marg_sum += float(-log_prior[tgt_len].sum())
                len_n += n_b

                target = batch["iface"].float()
                valid_if = batch["present"] & batch["node_mask"]
                if valid_if.any():
                    per_i = F.binary_cross_entropy_with_logits(
                        s["iface_logits"], target, reduction="none")
                    iface_sum += per_i[valid_if].sum().item()
                    iface_cnt += int(valid_if.sum().item())

                x0 = batch["x0"]
                B = x0.size(0)
                for tv in VAL_T_STEPS:
                    t = torch.full((B,), tv, device=DEVICE)
                    rand = torch.rand(x0.shape, dtype=torch.float32,
                                      device=DEVICE, generator=val_gen)
                    x_t, mask = q_sample(x0, t, rand=rand, return_mask=True)
                    logits = decoder(x_t, t, memory, memory_mask)
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
        n_len = max(len_n, 1)
        len_nll = len_nll_sum / n_len
        len_marg = len_marg_sum / n_len
        len_top5_rate = len_top5 / n_len
        len_mae = len_mae_sum / n_len
        iface_bce = iface_sum / max(iface_cnt, 1)
        detail = " ".join(f"t{tv}:{step_sum[tv]/step_cnt[tv]:.3f}"
                          for tv in VAL_T_STEPS if step_cnt[tv] > 0)
        if bad_this_epoch:
            print(f"  [注意] 本 epoch 跳过 {bad_this_epoch} 个非有限 loss 的 batch"
                  f"（累计 {n_bad}）")
        print(f"Val loss: {val_loss:.4f} | 界面 BCE: {iface_bce:.4f} | {detail}")
        print(f"  长度: NLL {len_nll:.4f} nats (边际基线 {len_marg:.4f}, "
              f"提升 {len_marg - len_nll:+.4f}) | top5 {len_top5_rate:.4f} | "
              f"E[k] MAE {len_mae:.3f}")

        # 采一条看看
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch(batch)
                s = struct_encoder(batch)
                k_probs = torch.softmax(length_predictor(batch["cond"][:1]), dim=-1)
                k = torch.multinomial(k_probs, 1).squeeze(1)
                samples = p_sample_loop(decoder, s["memory"][:1], PEPTIDE_LENGTH, k,
                                        memory_mask=s["memory_mask"][:1], device=DEVICE)
                print(f"Sampled (k={int(k.item())}, 真实 {int(batch['length'][0])}): "
                      f"{decode_peptide(samples[0], max_len=int(k.item()))}")
                break

        last_path = os.path.join(MODEL_SAVE_DIR, "model_last.pt")
        torch.save({
            "decoder": decoder.state_dict(),
            "length_predictor": length_predictor.state_dict(),
            "struct_encoder": struct_encoder.state_dict(),
        }, last_path)

        if val_loss < best_val_loss - EARLY_STOP_MIN_DELTA:
            best_val_loss = val_loss
            best_epoch = epoch
            best_path = os.path.join(MODEL_SAVE_DIR, "model_best.pt")
            torch.save({
                "decoder": decoder.state_dict(),
                "length_predictor": length_predictor.state_dict(),
                "struct_encoder": struct_encoder.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_t_steps": VAL_T_STEPS,
                "seed": SEED,
            }, best_path)
            patience_left = EARLY_STOP_PATIENCE
            print(f"[改进] 验证损失 {val_loss:.4f} 最优，已保存 {best_path}")
        else:
            patience_left -= 1
            print(f"验证损失未改进（最优 {best_val_loss:.4f} @ {best_epoch}，"
                  f"还等 {patience_left}/{EARLY_STOP_PATIENCE}）")
            if patience_left <= 0:
                stop_epoch = epoch
                print(f"连续 {EARLY_STOP_PATIENCE} 轮无改进，提前停止。")
                break

    if best_path is None:
        print("警告：没有保存任何 model_best.pt")
    else:
        print(f"训练结束：epoch {stop_epoch} 停 | 最优 epoch {best_epoch}，"
              f"验证损失 {best_val_loss:.4f} -> {best_path}")


if __name__ == "__main__":
    train()
