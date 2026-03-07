# Alignment: DPO & GRPO

LMT implements two post-training alignment methods for steering
language model behavior using human preferences or reward signals.

## Direct Preference Optimization (DPO)

*Rafailov et al., 2023*

DPO bypasses reward model training entirely. Given preference pairs
(chosen/rejected completions) and a frozen reference policy, it
optimizes the policy directly:

$$
\mathcal{L}_{DPO} = -\log\sigma\!\left(
    \beta \left[
        \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
      - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
    \right]
\right)
$$

**Key insight**: The optimal RLHF policy has a closed-form relationship
with the reward function, so you can skip the reward model entirely.

### Usage

```python
from lmt.training import dpo_loss, sequence_log_probs

# Get log-probs from policy and reference models
policy_logits = policy_model(input_ids)
ref_logits = ref_model(input_ids)

policy_chosen_logps = sequence_log_probs(policy_logits, chosen_ids)
policy_rejected_logps = sequence_log_probs(policy_logits, rejected_ids)
ref_chosen_logps = sequence_log_probs(ref_logits, chosen_ids)
ref_rejected_logps = sequence_log_probs(ref_logits, rejected_ids)

loss = dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta=0.1,  # KL penalty (higher = more conservative)
)
```

### Parameters

- **beta**: Controls how far the policy can drift from the reference.
  Higher values produce more conservative updates. Typical: 0.1-0.5.

---

## Group Relative Policy Optimization (GRPO)

*DeepSeek-AI, 2025 (DeepSeek-R1)*

GRPO eliminates the critic (value network) from PPO. For each prompt,
it generates a **group** of responses, scores them with a reward
function, and uses the group statistics as an implicit baseline:

$$
\hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}
                 {\text{std}(\mathbf{r})}
$$

$$
\mathcal{L}_{GRPO} = -\frac{1}{G} \sum_{i=1}^{G}
    \min\!\left(
        \rho_i \hat{A}_i,\;
        \text{clip}(\rho_i, 1\!-\!\epsilon, 1\!+\!\epsilon) \hat{A}_i
    \right)
    + \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})
$$

**Key insight**: By normalizing rewards within a group, you don't need
a learned value function -- the group mean serves as the baseline.
Especially effective for **verifiable tasks** (math, code) where
reward is binary or rule-based.

### Usage

```python
from lmt.training import grpo_loss, compute_group_advantages

# Generate G responses per prompt, score with reward function
rewards = reward_fn(responses)  # [group_size]

# Compute per-token log-probs for each response
policy_logps = ...   # [group_size, seq_len]
old_logps = ...      # from sampling policy (detached)
ref_logps = ...      # from frozen reference (detached)

loss = grpo_loss(
    policy_logps=policy_logps,
    old_logps=old_logps,
    ref_logps=ref_logps,
    rewards=rewards,
    clip_eps=0.2,    # PPO clipping
    kl_coeff=0.04,   # KL penalty
)
```

## DPO vs GRPO

| Aspect | DPO | GRPO |
|--------|-----|------|
| Input | Preference pairs | Scalar rewards |
| Reward model | None | None (rule-based) |
| Value network | None | None |
| Best for | Subjective preferences | Verifiable tasks |
| Complexity | ~20 lines | ~40 lines |
| From | Rafailov et al., 2023 | DeepSeek-R1, 2025 |

## References

- Rafailov et al., [*Direct Preference Optimization*](https://arxiv.org/abs/2305.18290) (2023)
- DeepSeek-AI, [*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*](https://arxiv.org/abs/2501.12948) (2025)
