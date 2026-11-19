"""Entropic advantage computation with adaptive beta (TTT-Discover, Section 3.2).

Ported from the reference implementation in
  discover-main/ttt_discover/rl/train.py  (compute_advantages, entropic_adaptive_beta).
"""

import math

import torch


def compute_entropic_advantages_adaptive_beta(
    rewards: torch.Tensor,
    gamma: float = math.log(2),
    beta_max: float = 1e6,
    bisection_iters: int = 60,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute entropic advantages with adaptive beta via bisection search.

    For a group of K rollouts with rewards r_1 ... r_K:
      1. Find beta such that KL(q_beta || uniform) = gamma,
         where q_beta(n) proportional to exp(beta * r_n).
      2. Compute LOO (Leave-One-Out) importance weights and advantages.

    Args:
        rewards: [K] tensor of scalar rewards for one group.
        gamma: target KL divergence between tilted and uniform distribution.
                Default ln(2) ~ 0.693 (1 bit of information).
        beta_max: upper bound for beta search.
        bisection_iters: number of bisection iterations.
        eps: numerical stability constant.

    Returns:
        advantages: [K] tensor.  Positive for above-average rollouts,
                    negative for below-average, zero when all rewards are equal.
    """
    r = rewards.float()
    k = r.shape[0]

    if k < 2:
        return torch.zeros_like(r)

    # If all rewards are identical, no signal
    if r.max() - r.min() < eps:
        return torch.zeros_like(r)

    log_k = math.log(k)
    r_max = r.max()

    # ----------------------------------------------------------------
    # Bisection to find beta such that KL(q_beta || uniform) = gamma
    # ----------------------------------------------------------------
    def kl_at_beta(beta_val: float) -> float:
        """KL(q_beta || uniform) where q_beta ∝ exp(beta * r)."""
        logits = beta_val * (r - r_max)  # shift for numerical stability
        log_q = logits - torch.logsumexp(logits, dim=0)
        q = torch.exp(log_q)
        # KL(q || u) = sum q_i * (log q_i - log(1/K)) = sum q_i * (log q_i + log K)
        kl = (q * (log_q + log_k)).sum()
        return float(kl.item())

    lo, hi = 0.0, 1.0

    # Expand hi until KL(hi) >= gamma
    while hi < beta_max and kl_at_beta(hi) < gamma:
        hi *= 2.0

    if kl_at_beta(hi) < gamma:
        # Could not reach target gamma; use best-effort beta
        beta_val = hi
    else:
        # Standard bisection
        for _ in range(bisection_iters):
            mid = 0.5 * (lo + hi)
            if kl_at_beta(mid) < gamma:
                lo = mid
            else:
                hi = mid
        beta_val = hi

    beta = torch.tensor(beta_val, dtype=r.dtype, device=r.device)

    # ----------------------------------------------------------------
    # LOO entropic advantages
    # ----------------------------------------------------------------
    e = torch.exp(beta * (r - r_max))  # [K], numerically stable

    # Leave-One-Out normaliser: Z_{-n} = (sum_m e_m - e_n) / (K - 1)
    e_sum = e.sum()
    z_loo = (e_sum - e) / (k - 1)

    # Importance weight and advantage
    w = e / (z_loo + eps)
    advantages = w - 1.0  # baseline = 1 since E[w] = 1

    return advantages
