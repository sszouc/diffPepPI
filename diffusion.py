import torch
import torch.nn.functional as F

from config import PAD_IDX, MASK_IDX, T_STEPS, X_IDX, EOS_IDX

def mask_schedule(t: torch.Tensor, T: int = T_STEPS) -> torch.Tensor:
    return t.float() / (T - 1)

def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    pad_idx: int = PAD_IDX,
    mask_idx: int = MASK_IDX,
    x_idx: int = X_IDX,
    T: int = T_STEPS,
) -> torch.Tensor:
    p_mask = mask_schedule(t, T).unsqueeze(1)
    rand = torch.rand_like(x0.float())

    mask = (rand < p_mask) & (x0 != pad_idx) & (x0 != x_idx)
    x_t = x0.clone()
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
    if device is None:
        device = cond.device

    model.eval()
    batch_size = cond.size(0)

    length = torch.as_tensor(length, dtype=torch.long, device=device).reshape(-1)

    pos = torch.arange(L, device=device).unsqueeze(0)
    k = length.unsqueeze(1)
    x_t = torch.where(pos < k, torch.full((1, 1), mask_idx, dtype=torch.long, device=device),
          torch.where(pos == k, torch.full((1, 1), eos_idx, dtype=torch.long, device=device),
                      torch.full((1, 1), pad_idx, dtype=torch.long, device=device))).long()

    with torch.no_grad():
        for t in range(T - 1, 0, -1):

            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            logits = model(x_t, t_tensor, cond)
            probs = F.softmax(logits, dim=-1)

            probs = probs.clone()
            probs[..., eos_idx] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            mask_pos = x_t.eq(mask_idx)
            if not mask_pos.any():
                continue

            p_t = t / (T - 1)
            p_prev = (t - 1) / (T - 1)
            unmask_prob = 1.0 - (p_prev / p_t)

            rand = torch.rand_like(x_t.float())
            to_unmask = mask_pos & (rand < unmask_prob)

            if to_unmask.any():

                probs_to_sample = probs[to_unmask]
                sampled = torch.multinomial(probs_to_sample, 1).squeeze(1)
                x_t[to_unmask] = sampled

        mask_pos = x_t.eq(mask_idx)
        if mask_pos.any():
            t_tensor = torch.zeros((batch_size,), device=device, dtype=torch.long)
            logits = model(x_t, t_tensor, cond)
            probs = F.softmax(logits, dim=-1).clone()
            probs[..., eos_idx] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            probs_to_sample = probs[mask_pos]
            sampled = torch.multinomial(probs_to_sample, 1).squeeze(1)
            x_t[mask_pos] = sampled

    return x_t
