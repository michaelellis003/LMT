r"""Direct Preference Optimization (DPO) loss.

Implements the preference learning objective from:
    Rafailov et al., 2023 -- "Direct Preference Optimization:
    Your Language Model is Secretly a Reward Model"

DPO bypasses reward model training entirely. Given a preference
pair (chosen response :math:`y_w`, rejected response :math:`y_l`)
and a frozen reference policy :math:`\pi_{ref}`, the loss is:

.. math::

    \mathcal{L}_{DPO} = -\log\sigma\!\left(
        \beta \left[
            \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
          - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
        \right]
    \right)

The key insight: the optimal RLHF policy satisfies a closed-form
relationship with the reward function, so we can optimize the policy
directly from preferences without ever learning a reward model.

:math:`\beta` controls how far the policy can drift from the
reference (higher = more conservative).
"""

import torch.nn.functional as f
from torch import Tensor


def sequence_log_probs(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    r"""Compute per-sequence sum of log-probabilities.

    Args:
        logits: Model output ``[batch, seq_len, vocab_size]``.
        labels: Target token IDs ``[batch, seq_len]``.
        mask: Optional boolean mask ``[batch, seq_len]`` where
            True means the position contributes to the sum.

    Returns:
        Log-probability sum per sequence ``[batch]``.
    """
    # Per-token log-probs: [batch, seq_len]
    log_probs = f.log_softmax(logits.float(), dim=-1)
    # Gather the log-prob of each target token
    per_token = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(
        -1
    )

    if mask is not None:
        per_token = per_token * mask.float()

    # Sum over sequence dimension -> [batch]
    return per_token.sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    ref_chosen_logps: Tensor,
    ref_rejected_logps: Tensor,
    beta: float = 0.1,
) -> Tensor:
    r"""Compute the DPO loss.

    Args:
        policy_chosen_logps: Log-prob of chosen responses under
            the policy ``[batch]``.
        policy_rejected_logps: Log-prob of rejected responses
            under the policy ``[batch]``.
        ref_chosen_logps: Log-prob of chosen responses under
            the frozen reference ``[batch]``.
        ref_rejected_logps: Log-prob of rejected responses
            under the frozen reference ``[batch]``.
        beta: KL penalty coefficient. Higher values keep the
            policy closer to the reference. Default: 0.1.

    Returns:
        Scalar DPO loss (mean over batch).
    """
    # Log-ratio advantages
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps

    # DPO: -log(sigma(beta * (chosen_adv - rejected_adv)))
    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -f.logsigmoid(logits).mean()

    return loss
