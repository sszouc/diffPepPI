#train.py
"""训练主循环：长度预测器与肽去噪网络【完全独立】的两阶段训练。

【每个 batch 的更新流程】
    ① 长度分支
         cond → LengthPredictor → 长度分布 → 与真实长度算交叉熵
         → len_loss.backward() → **只更新长度预测器的参数**
    ② 肽分支
         → q_sample 前向加噪得到 x_t → 去噪网络以 (x_t, t, cond) 为输入
         → 对有效位置（跳过 <PAD> 与未知氨基酸 <X>）算交叉熵
         → diff_loss.backward() → **只更新去噪网络的参数**

    两个分支各有自己的 AdamW 优化器、各自反传、互不干扰。
    注意：肽网络**不吃长度输入**——训练输入 x_t 里本来就能读出真实长度
    （位置 L 是 <EOS>、之后是永不被掩码的 <PAD>，还带 padding mask），
    实测把长度当条件喂进去时它只差 0.002 的损失，等于没用上，
    所以这一版把长度从网络输入里删掉了：预测长度只在**生成时用来定采样窗口**。

每轮结束后在验证集上报告：
    · 长度预测准确率（argmax 命中真实长度的比例）
    · 扩散损失（(x_t, t, cond) 下、每 token 平均）
再用验证集第一条样本做一次完整生成（采样长度 → 定窗口 → 生成 → 解码），最后存权重。

【与 PDF 附录原版的差异】
    1) 去掉混合精度与梯度累积（fp32、每个 step 直接更新）。
    2) 只取正样本训练（config.ONLY_POSITIVE）；未知氨基酸 <X> 被排除出损失。
    3) 词表扩展：新增 <EOS>（肽结束）与 <X>（未知，只作输入），输出类别 = 20 氨基酸 + EOS。
    4) 长度独立预测（不再"让模型自己在哪停"），且不作为网络输入，只用于定采样窗口。
"""
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
    X_IDX,              # 未知氨基酸：需要从损失里排除
    MODEL_SAVE_DIR,
    PEPTIDE_LENGTH,
    SEED,
)
from data_utils import get_dataloaders, set_seed, decode_peptide
from diffusion import q_sample, p_sample_loop
from model import ConditionalDecoder, LengthPredictor


def train() -> None:
    """训练入口：没有任何命令行参数，全部超参数取自 config.py。

    想在实验中途改变轮数 / 批量 / 学习率，直接改 config.py 再重跑即可。
    """
    set_seed(SEED)                                    # 固定种子
    train_loader, val_loader = get_dataloaders(BATCH_SIZE)

    decoder = ConditionalDecoder().to(DEVICE)         # 肽去噪网络（生成内容）
    length_predictor = LengthPredictor().to(DEVICE)   # 长度预测器（决定多长）
    # 两个网络各用各的优化器：这一步保证了"两阶段"互不干扰
    opt_dec = AdamW(decoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    opt_len = AdamW(length_predictor.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    best_ckpt = None
    for epoch in range(1, EPOCHS + 1):               # epoch 从 1 开始，方便拼权重文件名
        decoder.train()
        length_predictor.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}")

        # ★ 每次迭代拿到的是**一个 batch**（BATCH_SIZE = 64 条记录），不是一条记录：
        #   DataLoader(dataset, batch_size=64) 先按索引取出 64 个单条样本再堆叠，于是：
        #       cond   : (64, EMBEDDING_DIM)  第 i 行 = 第 i 条肽对应的蛋白嵌入
        #       x0     : (64, PEPTIDE_LENGTH) 第 i 行 = 第 i 条肽的 51 个位置（肽 + EOS + PAD）
        #       length : (64,)                第 i 个 = 第 i 条肽的真实长度
        #   一个 epoch 迭代 ceil(7759 / 64) = 122 次（最后一个 batch 只有 15 条）。
        for cond, x0, length in progress:
            cond = cond.to(DEVICE)     # 蛋白质嵌入向量，形状 (B, 1280)，B = 64
            x0 = x0.to(DEVICE)         # 肽索引序列，形状 (B, 51)，B = 64
            length = length.to(DEVICE) # 真实肽长度 (B,)：长度分支的标签

            # ================= ① 长度分支：只更新长度预测器 =================
            len_logits = length_predictor(cond)              # (B, 51) 长度 0..50 的打分
            len_loss = F.cross_entropy(len_logits, length)   # 与"真实长度的距离"（交叉熵）
            opt_len.zero_grad(set_to_none=True)
            len_loss.backward()                              # 只回传长度预测器的梯度
            opt_len.step()

            # ================= ② 肽分支：只更新去噪网络 =================
            # 注意：这里不再喂长度——去噪网络只吃 (x_t, t, cond)。
            # 每条样本独立采一个时间步；randint 的上界是开区间，故写 T_SAMPLE_MAX + 1
            t = torch.randint(
                T_SAMPLE_MIN,
                T_SAMPLE_MAX + 1,
                (x0.size(0),),
                device=DEVICE,
            )
            x_t = q_sample(x0, t)                            # 前向加噪：(B, L)，MASK 位置为 22

            logits = decoder(x_t, t, cond)                   # (B, L, 21) = 20 氨基酸 + EOS
            logits_flat = logits.view(-1, logits.size(-1))   # (B*L, 21)
            target_flat = x0.view(-1)                        # (B*L,)  目标是干净序列
            # ★ 把未知氨基酸 <X> 也改成 <PAD>：cross_entropy 只能忽略一个索引，
            #   而我们要同时忽略"补齐位 PAD"和"未知位 X"，于是先做一次等价替换，
            #   再统一用 ignore_index=PAD_IDX 屏蔽掉这两类位置。
            target_flat = target_flat.masked_fill(target_flat == X_IDX, PAD_IDX)
            diff_loss = F.cross_entropy(
                logits_flat,
                target_flat,
                reduction="mean",        # 对"有效位置"取平均
                ignore_index=PAD_IDX,    # PAD（含被改成 PAD 的 X）不贡献损失也不贡献分母
            )
            opt_dec.zero_grad(set_to_none=True)
            diff_loss.backward()                             # 只回传去噪网络的梯度
            opt_dec.step()

            # 进度条上同时看长度损失、长度准确率、内容损失
            with torch.no_grad():
                len_acc = (len_logits.argmax(dim=-1) == length).float().mean().item()
            progress.set_postfix(len=f"{len_loss.item():.3f}",
                                 acc=f"{len_acc:.2f}",
                                 diff=f"{diff_loss.item():.3f}")

        # ---------------- 验证：同一套加噪流程，但只前向 ----------------
        decoder.eval()
        length_predictor.eval()
        val_sum, val_tokens = 0.0, 0
        len_correct, len_total = 0, 0
        with torch.no_grad():
            for cond, x0, length in val_loader:
                cond = cond.to(DEVICE)
                x0 = x0.to(DEVICE)
                length = length.to(DEVICE)

                # 长度分支：只算准确率（不再是网络输入，所以不影响下面的扩散损失）
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
                # 与训练同样：用 reduction="none" 保留每个位置的损失，
                # 再按有效位置（非 PAD、非 X）加权平均，得到"每 token 平均损失"
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

        # ---------------- 每轮做一次完整生成，检查流程有没有跑偏 ----------------
        # 这里体现"先预测长度、再定窗口"：k 用来决定采样 0..k-1 与在哪放 <EOS>/<PAD>，
        # 但**不喂给网络**（网络只看 x_t / t / cond）
        # Validation sampling on first batch
        with torch.no_grad():
            for cond, _, length in val_loader:                # 只取验证集的第一个 batch
                cond = cond.to(DEVICE)
                k_probs = torch.softmax(length_predictor(cond[:1]), dim=-1)   # (1, 51)
                k = torch.multinomial(k_probs, 1).squeeze(1)                  # 采样长度
                samples = p_sample_loop(decoder, cond[:1], PEPTIDE_LENGTH, k, device=DEVICE)
                peptide = decode_peptide(samples[0], max_len=int(k.item()))   # 完整跑 500 步去噪
                print(f"Sampled peptide (采样长度 {int(k.item())}, 真实长度 {int(length[0])}): {peptide}")
                break

        # ---------------- 保存权重 ----------------
        # 每轮都存一个文件；两个网络一起存成字典 {"decoder":..., "length_predictor":...}
        ckpt_path = os.path.join(MODEL_SAVE_DIR, f"model_epoch_{epoch}.pt")
        torch.save(
            {"decoder": decoder.state_dict(),
             "length_predictor": length_predictor.state_dict()},
            ckpt_path,
        )
        print(f"Saved checkpoint: {ckpt_path}")

        # 以验证损失挑选最优轮次
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
