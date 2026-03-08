r"""Group Relative Policy Optimization (GRPO).

Implements the RL objective from:
    DeepSeek-AI, 2025 -- "DeepSeek-R1: Incentivizing Reasoning
    Capability in LLMs via Reinforcement Learning"

GRPO eliminates the critic (value network) entirely. For each
prompt, it generates a **group** of G responses, scores them with
a reward function (e.g., correctness checker), normalizes rewards
within the group to get advantages, and applies a PPO-style
clipped objective.

.. math::

    \hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}
                     {\text{std}(\mathbf{r}) + \epsilon}

.. math::

    \mathcal{L}_{GRPO} = -\frac{1}{G} \sum_{i=1}^{G}
        \min\!\left(
            \rho_i \hat{A}_i,\;
            \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) \hat{A}_i
        \right)
        + \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})

where :math:`\rho_i = \frac{\pi_\theta(y_i|x)}{\pi_{old}(y_i|x)}`
is the importance sampling ratio.

Key insight: by normalizing rewards within a group, GRPO doesn't
need a learned value function -- the group mean serves as an
implicit baseline. This is especially effective for verifiable
tasks (math, code) where reward is binary or rule-based.
"""

import torch
from torch import Tensor


def compute_group_advantages(
    rewards: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Normalize rewards within a group to get advantages.

    Args:
        rewards: Reward scores ``[group_size]``.
        eps: Epsilon for numerical stability in std.

    Returns:
        Normalized advantages ``[group_size]`` with ~zero mean
        and ~unit variance.
    """
    if rewards.numel() <= 1:
        return torch.zeros_like(rewards)

    mean = rewards.mean()
    std = rewards.std()

    if std < eps:
        return torch.zeros_like(rewards)

    return (rewards - mean) / (std + eps)


def grpo_loss(
    policy_logps: Tensor,
    old_logps: Tensor,
    ref_logps: Tensor,
    rewards: Tensor,
    clip_eps: float = 0.2,
    kl_coeff: float = 0.04,
) -> Tensor:
    r"""Compute the GRPO loss.

    Args:
        policy_logps: Per-token log-probs under current policy
            ``[group_size, seq_len]``.
        old_logps: Per-token log-probs under the sampling policy
            ``[group_size, seq_len]``. Detached (no grad).
        ref_logps: Per-token log-probs under the frozen reference
            ``[group_size, seq_len]``. Detached (no grad).
        rewards: Reward scores for each response ``[group_size]``.
        clip_eps: PPO clipping epsilon. Default: 0.2.
        kl_coeff: KL penalty coefficient (beta). Default: 0.04.

    Returns:
        Scalar GRPO loss.
    """
    # Group-normalized advantages: [group_size]
    advantages = compute_group_advantages(rewards)

    # Per-token importance ratios: [group_size, seq_len]
    log_ratios = policy_logps - old_logps.detach()
    ratios = log_ratios.exp()

    # Expand advantages to [group_size, 1] for broadcasting
    adv = advantages.unsqueeze(-1)

    # Clipped surrogate objective (PPO-style)
    surr1 = ratios * adv
    surr2 = ratios.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()

    # KL penalty: approximate KL(pi_theta || pi_ref) per token
    # Using the unbiased estimator: exp(log_ratio) - log_ratio - 1
    kl_log_ratios = policy_logps - ref_logps.detach()
    kl = (kl_log_ratios.exp() - kl_log_ratios - 1.0).mean()

    return policy_loss + kl_coeff * kl
