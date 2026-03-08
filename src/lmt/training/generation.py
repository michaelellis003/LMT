"""Batch response generation for GRPO training.

Generates groups of responses for a given prompt using temperature
sampling with top-k and top-p (nucleus) filtering. Supports stop
token handling with padding for uniform-length output tensors.

This module is designed for GRPO's "generate G responses per prompt"
step. The generation runs in eval mode with no gradient tracking.

Usage::

    from lmt.training.generation import GenerationConfig, generate_responses

    config = GenerationConfig(group_size=8, max_response_len=128)
    sequences = generate_responses(model, prompt_ids, config)
    # sequences: [8, prompt_len + 128]
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as f


@dataclass
class GenerationConfig:
    """Configuration for batch response generation.

    Attributes:
        group_size: Number of responses to generate per prompt (G).
        max_response_len: Maximum tokens to generate per response.
        temperature: Sampling temperature. Lower = more deterministic.
        top_k: Top-k filtering. 0 = disabled.
        top_p: Top-p (nucleus) filtering. 1.0 = disabled.
        stop_token_id: If set, stop generating when this token appears
            and pad remaining positions.
        pad_token_id: Token ID used for padding after stop token.
    """

    group_size: int = 8
    max_response_len: int = 128
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 1.0
    stop_token_id: int | None = None
    pad_token_id: int = 0


def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Zero out logits below the top-k threshold.

    Args:
        logits: Raw logits ``[batch, vocab_size]``.
        top_k: Number of top tokens to keep.

    Returns:
        Filtered logits with non-top-k positions set to ``-inf``.
    """
    top_k = min(top_k, logits.size(-1))
    values, _ = logits.topk(top_k)
    min_val = values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < min_val, float('-inf'))


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply nucleus (top-p) filtering.

    Keeps the smallest set of tokens whose cumulative probability
    exceeds ``top_p``. All other tokens are set to ``-inf``.

    Args:
        logits: Raw logits ``[batch, vocab_size]``.
        top_p: Cumulative probability threshold (0.0 to 1.0).

    Returns:
        Filtered logits.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(f.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    # Shift right so the first token above threshold is kept
    sorted_mask = cumulative_probs - f.softmax(sorted_logits, dim=-1)
    sorted_mask = sorted_mask > top_p

    # Scatter the mask back to original positions
    mask = sorted_mask.scatter(dim=-1, index=sorted_indices, src=sorted_mask)
    return logits.masked_fill(mask, float('-inf'))


@torch.no_grad()
def generate_responses(
    model: nn.Module,
    prompt: torch.Tensor,
    config: GenerationConfig,
) -> torch.Tensor:
    """Generate a group of responses for a prompt.

    Runs the model in eval mode (restored after generation) and
    generates ``config.group_size`` responses using autoregressive
    sampling with temperature, top-k, and top-p filtering.

    Args:
        model: Language model that accepts token IDs and returns
            logits ``[batch, seq_len, vocab_size]``.
        prompt: Prompt token IDs ``[prompt_len]`` (1D).
        config: Generation configuration.

    Returns:
        Full sequences (prompt + response) of shape
        ``[group_size, prompt_len + max_response_len]``.
    """
    was_training = model.training
    model.eval()

    device = next(model.parameters()).device
    prompt_len = prompt.shape[0]

    # Expand prompt to [group_size, prompt_len]
    sequences = (
        prompt.to(device).unsqueeze(0).expand(config.group_size, -1).clone()
    )

    # Track which sequences have hit stop token
    finished = torch.zeros(config.group_size, dtype=torch.bool, device=device)

    # Get context length from model config
    model_config = getattr(model, 'config', None)
    ctx_len = getattr(model_config, 'context_length', None)

    for _ in range(config.max_response_len):
        # Truncate to context window if needed
        inputs = sequences[:, -ctx_len:] if ctx_len is not None else sequences

        logits = model(inputs)[:, -1, :]  # [G, vocab]

        # Temperature scaling
        if config.temperature != 1.0:
            logits = logits / config.temperature

        # Top-k filtering
        if config.top_k > 0:
            logits = _apply_top_k(logits, config.top_k)

        # Top-p filtering
        if config.top_p < 1.0:
            logits = _apply_top_p(logits, config.top_p)

        probs = f.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)  # [G, 1]

        # Replace tokens for finished sequences with pad
        if config.stop_token_id is not None:
            next_tokens = next_tokens.masked_fill(
                finished.unsqueeze(-1), config.pad_token_id
            )

        sequences = torch.cat([sequences, next_tokens], dim=1)

        # Check for stop token
        if config.stop_token_id is not None:
            just_stopped = next_tokens.squeeze(-1) == config.stop_token_id
            finished = finished | just_stopped

            # Early exit if all sequences are done
            if finished.all():
                # Pad remaining positions
                remaining = config.max_response_len - (
                    sequences.shape[1] - prompt_len
                )
                if remaining > 0:
                    pad = torch.full(
                        (config.group_size, remaining),
                        config.pad_token_id,
                        dtype=sequences.dtype,
                        device=device,
                    )
                    sequences = torch.cat([sequences, pad], dim=1)
                break

    if was_training:
        model.train()

    return sequences
