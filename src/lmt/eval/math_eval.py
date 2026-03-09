"""Math reasoning evaluation for LMT models.

Evaluates a model's ability to solve math problems by generating
responses and checking if the extracted answer matches the ground truth.

Usage::

    from lmt.eval.math_eval import evaluate_math_accuracy

    result = evaluate_math_accuracy(
        model=model,
        items=math_items,
        tokenizer=tokenizer,
        max_new_tokens=256,
    )
    print(f'Accuracy: {result["accuracy"]:.1%}')
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from lmt.data.math_data import MathDataItem, extract_boxed_answer
from lmt.training.rewards import math_reward


def evaluate_math_accuracy(
    model: nn.Module,
    items: list[MathDataItem],
    tokenizer: Any,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Evaluate math accuracy by generating and checking answers.

    For each math item, encodes the prompt, generates a response
    using greedy decoding (or sampling), extracts the boxed answer,
    and compares with the ground truth.

    Args:
        model: The language model to evaluate.
        items: List of math problems with ground truth answers.
        tokenizer: Tokenizer with ``encode`` and ``decode`` methods.
        max_new_tokens: Maximum tokens to generate per problem.
        temperature: Sampling temperature (0.0 = greedy).

    Returns:
        Dict with keys ``accuracy``, ``num_correct``, ``num_total``,
        and ``details`` (list of per-item results).
    """
    if not items:
        return {
            'accuracy': 0.0,
            'num_correct': 0,
            'num_total': 0,
            'details': [],
        }

    model.eval()
    device = next(model.parameters()).device

    num_correct = 0
    details: list[dict[str, Any]] = []

    for item in items:
        prompt_ids = tokenizer.encode(item.prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # Generate response
        with torch.no_grad():
            generated = _greedy_generate(
                model, input_ids, max_new_tokens, temperature
            )

        # Extract response tokens (after prompt)
        response_ids = generated[0, len(prompt_ids) :].tolist()
        response_text = tokenizer.decode(response_ids)

        # Check answer
        reward = math_reward(
            prompt=item.prompt,
            response=response_text,
            ground_truth=item.ground_truth,
        )
        correct = reward == 1.0
        if correct:
            num_correct += 1

        extracted = extract_boxed_answer(response_text)
        details.append(
            {
                'prompt': item.prompt,
                'ground_truth': item.ground_truth,
                'response': response_text,
                'extracted_answer': extracted,
                'correct': correct,
            }
        )

    accuracy = num_correct / len(items)
    return {
        'accuracy': accuracy,
        'num_correct': num_correct,
        'num_total': len(items),
        'details': details,
    }


def _greedy_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
) -> torch.Tensor:
    """Generate tokens greedily (or with temperature sampling).

    Args:
        model: Language model that returns logits.
        input_ids: Input token IDs [1, seq_len].
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature. 0.0 = argmax.

    Returns:
        Generated sequence [1, seq_len + new_tokens].
    """
    generated = input_ids
    config = getattr(model, 'config', None)
    ctx_len = getattr(config, 'context_length', 2048) if config else 2048

    for _ in range(max_new_tokens):
        # Truncate to context length
        inp = generated[:, -ctx_len:]
        logits = model(inp)
        next_logits = logits[:, -1, :]  # Last position

        if temperature <= 0.0:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)

    return generated
