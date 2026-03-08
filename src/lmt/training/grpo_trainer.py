r"""GRPO training loop.

Orchestrates Group Relative Policy Optimization:
1. Generate G responses per prompt (group sampling)
2. Score each response with a reward function
3. Compute group-normalized advantages
4. Apply PPO-style clipped objective with optional KL penalty

Follows Raschka's "reasoning-from-scratch" recipe:
- No SFT step required (GRPO directly on base model)
- **No KL penalty by default** (kl_coeff=0.0) -- Raschka found
  that the KL term causes training collapse at small scale
- Group size G=8 is sufficient
- 50 training steps can show significant improvement

References:
    DeepSeek-AI (2025). "DeepSeek-R1"
    Raschka, S. (2025). "Reasoning from Scratch"
"""

import copy
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as f

from lmt.training.grpo import grpo_loss


@dataclass
class GRPOConfig:
    """Configuration for GRPO training.

    Attributes:
        group_size: Number of responses per prompt (G).
        max_response_len: Maximum tokens to generate per response.
        clip_eps: PPO clipping epsilon.
        kl_coeff: KL penalty coefficient. Default 0.0 (disabled)
            following Raschka's finding that KL hurts at small scale.
        lr: Learning rate.
        temperature: Sampling temperature for response generation.
        top_k: Top-k filtering for sampling. 0 = disabled.
        device: Device for training.
    """

    group_size: int = 8
    max_response_len: int = 128
    clip_eps: float = 0.2
    kl_coeff: float = 0.0
    lr: float = 5e-6
    max_grad_norm: float | None = 1.0
    temperature: float = 0.7
    top_k: int = 50
    device: str = 'cpu'


# Type alias for reward functions
RewardFn = Callable[[torch.Tensor, torch.Tensor], float]


class GRPOTrainer:
    """GRPO training loop for reinforcement learning with verifiable rewards.

    Args:
        model: The policy model to train.
        config: GRPO training configuration.
        ref_model: Optional frozen reference model. If None, a deep
            copy of the policy model is created and frozen.
    """

    def __init__(
        self,
        model: nn.Module,
        config: GRPOConfig,
        ref_model: nn.Module | None = None,
    ) -> None:
        """Initialize GRPO trainer.

        Args:
            model: Policy model to train.
            config: Training configuration.
            ref_model: Frozen reference model. If None, creates a
                deep copy of the policy model.
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device)

        # Create frozen reference model
        if ref_model is not None:
            self.ref_model = ref_model
        else:
            self.ref_model = copy.deepcopy(model)

        # Freeze reference model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

        # Training state
        self.global_step = 0
        self.train_losses: list[float] = []

    def compute_log_probs(
        self,
        model: nn.Module,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token log-probs for a sequence.

        Args:
            model: Model to compute log-probs with.
            token_ids: Token IDs ``[batch, seq_len]``.

        Returns:
            Per-token log-probs ``[batch, seq_len - 1]`` for the
            next-token prediction at each position.
        """
        # Input is all tokens except last, target is all except first
        inputs = token_ids[:, :-1]
        targets = token_ids[:, 1:]

        logits = model(inputs)  # [batch, seq_len-1, vocab]
        # Upcast to FP32 for numerical stability (codebase convention)
        log_probs = f.log_softmax(logits.float(), dim=-1)

        # Gather log-probs for actual next tokens
        target_log_probs = log_probs.gather(
            dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1)

        return target_log_probs

    @torch.no_grad()
    def _generate_responses(
        self,
        prompt: torch.Tensor,
    ) -> torch.Tensor:
        """Generate G responses for a prompt using the current policy.

        Args:
            prompt: Prompt token IDs ``[prompt_len]``.

        Returns:
            Full sequences (prompt + response) ``[G, total_len]``.
        """
        self.model.eval()

        group_size = self.config.group_size
        prompt_len = prompt.shape[0]
        max_len = prompt_len + self.config.max_response_len

        # Expand prompt to [group_size, prompt_len]
        sequences = (
            prompt.to(self.device).unsqueeze(0).expand(group_size, -1).clone()
        )

        config = getattr(self.model, 'config', None)

        for _ in range(self.config.max_response_len):
            # Get context window
            ctx_len = getattr(config, 'context_length', max_len)
            inputs = sequences[:, -ctx_len:]

            logits = self.model(inputs)[:, -1, :]  # [G, vocab]

            # Temperature scaling
            if self.config.temperature != 1.0:
                logits = logits / self.config.temperature

            # Top-k filtering
            if self.config.top_k > 0:
                top_k = min(self.config.top_k, logits.size(-1))
                values, _ = logits.topk(top_k)
                min_val = values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < min_val, float('-inf'))

            probs = f.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            sequences = torch.cat([sequences, next_tokens], dim=1)

        self.model.train()
        return sequences

    def train_step(
        self,
        prompt: torch.Tensor,
        reward_fn: RewardFn,
    ) -> float:
        """Execute one GRPO training step for a single prompt.

        1. Generate G responses
        2. Score with reward function
        3. Compute log-probs under policy, old policy, and ref
        4. Compute GRPO loss (response tokens only) and update

        Args:
            prompt: Prompt token IDs ``[prompt_len]``.
            reward_fn: Callable(prompt_ids, response_ids) -> float.

        Returns:
            Loss value as float.
        """
        self.model.train()

        # Step 1: Generate responses (no grad)
        sequences = self._generate_responses(prompt)
        prompt_len = prompt.shape[0]

        # Step 2: Score responses
        rewards = torch.tensor(
            [reward_fn(prompt, seq[prompt_len:]) for seq in sequences],
            dtype=torch.float32,
            device=self.device,
        )

        # Step 3: Compute log-probs over FULL sequences
        with torch.no_grad():
            old_logps_full = self.compute_log_probs(self.model, sequences)
            ref_logps_full = self.compute_log_probs(self.ref_model, sequences)

        policy_logps_full = self.compute_log_probs(self.model, sequences)

        # CRITICAL: Only use response tokens for GRPO loss.
        # compute_log_probs returns [batch, seq_len-1], and prompt
        # occupies positions 0..prompt_len-1 in the original sequence.
        # After the shift in compute_log_probs, response predictions
        # start at index (prompt_len - 1).
        resp_start = prompt_len - 1
        policy_logps = policy_logps_full[:, resp_start:]
        old_logps = old_logps_full[:, resp_start:]
        ref_logps = ref_logps_full[:, resp_start:]

        # Step 4: Compute GRPO loss and update
        loss = grpo_loss(
            policy_logps=policy_logps,
            old_logps=old_logps,
            ref_logps=ref_logps,
            rewards=rewards,
            clip_eps=self.config.clip_eps,
            kl_coeff=self.config.kl_coeff,
        )

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
        self.optimizer.step()

        loss_val = loss.item()
        self.train_losses.append(loss_val)
        self.global_step += 1

        return loss_val
