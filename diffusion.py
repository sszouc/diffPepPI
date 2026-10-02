#diffusion.py
"""吸收态离散扩散（masked / absorbing diffusion）的前向加噪与反向采样。

核心思想：把肽序列看成 L 个互相独立的离散位置（L = config.PEPTIDE_LENGTH，现为 51），
前向过程按时间步 t 把每个位置以概率 p_t 替换成 <MASK>（吸收态：一旦变 MASK 就不会再
变回别的 token）；反向过程训练一个网络，让它看到“带 MASK 的含噪序列 + 时间步 + 蛋白
条件”后，直接预测每个位置**原始（干净）的 token**，其中既包括 20 种氨基酸，
也包括表示“肽到此结束”的 <EOS>。

定义在整个文件里通用的记号：
    T           扩散总步数（config.T_STEPS = 500）
    t           当前时间步，取值 0..T-1；t = 0 表示完全干净，t = T-1 表示全部 MASK
    p_t         掩码概率（mask_schedule），本实现取线性调度 p_t = t / (T - 1)
    x0          干净序列索引 (B, L)，取值 0..20（氨基酸或 EOS）、21 = PAD、23 = X
    x_t         第 t 步的含噪序列索引 (B, L)，取值 0..20 / 21(PAD) / 22(MASK) / 23(X)

时间步的两端约定：
    t = T-1 = 499 → p_t = 499/499 = 1.0 → 全 MASK（采样起点）
    t = 0        → p_t = 0             → 完全干净
"""
import torch
import torch.nn.functional as F

from config import PAD_IDX, MASK_IDX, T_STEPS, X_IDX, EOS_IDX


def mask_schedule(t: torch.Tensor, T: int = T_STEPS) -> torch.Tensor:
    """线性掩码调度：返回每个样本在时间步 t 的掩码概率 p_t = t / (T - 1)。

    参数：
        t : (B,) long 或 float 的时间步张量
        T : 扩散总步数

    返回：
        (B,) float，取值 [0, 1]；t 越大噪声越重。

    说明：这里用 (T - 1) 而不是 T 作分母，是为了让 t = T-1 时恰好 p_t = 1.0
    （保证采样能从“完全被 MASK 覆盖”的状态出发）；相应地 t = 0 时 p_t = 0。
    """
    return t.float() / (T - 1)


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    pad_idx: int = PAD_IDX,
    mask_idx: int = MASK_IDX,
    x_idx: int = X_IDX,
    T: int = T_STEPS,
) -> torch.Tensor:
    """前向加噪 q(x_t | x_0)：把干净序列 x0 按时间步 t 随机掩码，得到 x_t。

    参数：
        x0 : (B, L) long —— 干净肽索引（0..20 = 氨基酸/EOS，21 = PAD，23 = X）
        t  : (B,) long   —— 每个样本各自的时间步（同一条 batch 里 t 可以不同）

    做法（逐元素独立，不需要循环 step 逐步加噪，因为吸收态掩码有闭式形式）：
        1. 位置掩码条件：rand < p_t；rand 是均匀分布 U(0,1) 的随机数，
           因此每个位置被选中的概率恰好是 p_t；
        2. 但 <PAD> 位永远不掩码（补齐用的假残基，不该让模型去学恢复它）；
        3. 未知氨基酸 <X> 也**不掩码**——它是数据里本来就存在的“未知”信息，
           保持原样让模型知道“这个位置未知”，比把它变成 MASK 更诚实
           （这些位置的损失会在 train.py 里被排除掉）；
        4. 被选中的位置写为 <MASK>（索引 22）。
    返回：
        x_t : (B, L) long —— 取值 0..20 / 21(PAD) / 22(MASK) / 23(X)

    形状细节：mask_schedule(t) 是 (B,)，unsqueeze(1) 后变成 (B, 1)，
    与 (B, L) 的 rand / x0 做广播比较。
    """
    p_mask = mask_schedule(t, T).unsqueeze(1)          # (B, 1)
    rand = torch.rand_like(x0.float())                 # (B, L) ~ U(0, 1)
    # 只是和x0形状相同，和x0具体的内容无关
    mask = (rand < p_mask) & (x0 != pad_idx) & (x0 != x_idx)   # (B, L) bool
    x_t = x0.clone()                                   # 不原地修改调用方的 x0
    x_t[mask] = mask_idx
    return x_t


def p_sample_loop(
    model,
    cond: torch.Tensor,
    L: int,
    length: torch.Tensor,
    T: int = T_STEPS,
    pad_idx: int = PAD_IDX,
    mask_idx: int = MASK_IDX,
    eos_idx: int = EOS_IDX,
    device: torch.device | None = None,
) -> torch.Tensor:
    """按**给定长度**做反向采样：为每条样本生成恰好 length[i] 个残基。

    参数：
        model  : ConditionalDecoder，输入 (x_t, t, cond) 输出 (B, L, OUT_CLASSES=21)
        cond   : (B, EMBEDDING_DIM) 蛋白条件
        L      : 窗口位置数（= config.PEPTIDE_LENGTH = 51，即“最长肽 50 + EOS”）
        length : (B,) long 每条样本的目标长度 k（来自 LengthPredictor 的分布采样）。
                 注意：k **不喂给网络**，只用来决定窗口形状（哪里是待采样位、哪里放 EOS/PAD）。
        device : 计算设备；None 时跟随 cond 所在设备

    返回：
        x_t : (B, L) long，取值 0..20（20 种氨基酸或 <EOS>），不会再含 MASK

    【长度是怎么定下来的】
        本函数不再"让模型自己决定在哪停"，而是**先由长度预测器给出 k**，再按 k 构造窗口：
            位置 0..k-1 : MASK  → 需要采样出真实残基
            位置 k      : <EOS> → 长度已定，直接放终止符（不参与采样）
            位置 k+1..  : <PAD> → 完全不参与
        于是 <EOS> 永远不会被"自由采样"出来，文献里那个 `<eos>` overflow
        （EOS 既是终止符又是填充符 → 概率质量堆积 → 过早终止）在这里根本不会发生。
        解码时用 data_utils.decode_peptide(x, max_len=k)，输出恰好 k 个残基。

    迭代流程（t 从 T-1 递减到 1）：
        1. 用当前 x_t、时间步 t、条件 cond、长度 k 前向，得到每个位置对 21 个类别的分布；
        2. 计算“本步该把哪些 MASK 位置解码出来”的概率：
               unmask_prob = 1 - p_{t-1} / p_t
           推导：吸收态下 P(x_{t-1} = 真实值 | x_t = MASK) = 1 - p_{t-1} / p_t
           （即“在 t 时刻仍被遮住的前提下，它其实是在 t-1 就被解开的概率”），
           所以这是吸收态后验的精确形式，而不是经验启发式；
        3. 从当前所有 MASK 位里按该概率抽一部分位置解锁，解锁位置的取值
           从模型预测的多项分布中采样（torch.multinomial）；
        4. 已不是 MASK 的位置（含末尾的 EOS/PAD）在后续步骤中保持不变。

    收尾：如果循环结束时仍有 MASK（当 T 较小时可能发生），用 t = 0（干净）再过一次
    网络，把所有剩余 MASK 一次性填成采样值，保证返回的序列干净可解码。

    注意：整个函数在 torch.no_grad() 下运行，并且会先调 model.eval() 关掉 dropout。
    """
    if device is None:
        device = cond.device

    model.eval()
    batch_size = cond.size(0)
    # 长度统一成 (B,) long 张量，便于下面的向量化构造
    length = torch.as_tensor(length, dtype=torch.long, device=device).reshape(-1)
    # 向量化构造初始状态：(B, L)
    #   pos <  k → MASK（待采样） ; pos == k → EOS（终止符，直接放好） ; pos > k → PAD
    pos = torch.arange(L, device=device).unsqueeze(0)            # (1, L)
    k = length.unsqueeze(1)                                      # (B, 1)
    x_t = torch.where(pos < k, torch.full((1, 1), mask_idx, dtype=torch.long, device=device),
          torch.where(pos == k, torch.full((1, 1), eos_idx, dtype=torch.long, device=device),
                      torch.full((1, 1), pad_idx, dtype=torch.long, device=device))).long()

    with torch.no_grad():
        for t in range(T - 1, 0, -1):                  # t = 499, 498, ..., 1
            # 同一条 batch 里所有样本共用同一个时间步 t
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            logits = model(x_t, t_tensor, cond)          # (B, L, 21) = 20 氨基酸 + EOS
            probs = F.softmax(logits, dim=-1)          # 逐位置归一化，得到氨基酸概率
            # 【长度已定】待采样的位置是 0..k-1，这里一律禁止再采出 <EOS>：
            # 把 EOS 的概率清零再重新归一化，EOS 只保留在我们预先放好的位置 k 上，
            # 于是解码长度一定等于预测长度（也避免模型自己"提前收尾"）。
            probs = probs.clone()
            probs[..., eos_idx] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            mask_pos = x_t.eq(mask_idx)                # (B, L) 当前还是 MASK 的位置
            if not mask_pos.any():                     # 已全部解码完 → 提前结束
                continue

            p_t = t / (T - 1)                          # 当前步的掩码概率
            p_prev = (t - 1) / (T - 1)                 # 上一步的掩码概率
            unmask_prob = 1.0 - (p_prev / p_t)         # 本步应解锁的比例（吸收态后验）

            rand = torch.rand_like(x_t.float())        # (B, L) ~ U(0, 1)
            to_unmask = mask_pos & (rand < unmask_prob)   # 本步真正解锁的位置

            if to_unmask.any():
                # 只对解锁位置取分布：(N, 20)，N = 本次解锁的位置数
                probs_to_sample = probs[to_unmask]
                sampled = torch.multinomial(probs_to_sample, 1).squeeze(1)  # (N,)
                x_t[to_unmask] = sampled           # 布尔索引赋值，写回对应位置

        # ---- 兜底：把可能残留的 MASK 用 t = 0 再解一次，保证输出干净 ----
        mask_pos = x_t.eq(mask_idx)
        if mask_pos.any():
            t_tensor = torch.zeros((batch_size,), device=device, dtype=torch.long)
            logits = model(x_t, t_tensor, cond)
            probs = F.softmax(logits, dim=-1).clone()
            probs[..., eos_idx] = 0.0                  # 同样不允许采出 EOS（长度已定）
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            probs_to_sample = probs[mask_pos]
            sampled = torch.multinomial(probs_to_sample, 1).squeeze(1)
            x_t[mask_pos] = sampled

    return x_t
