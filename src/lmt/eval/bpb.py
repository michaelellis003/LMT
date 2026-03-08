r"""Bits Per Byte (BPB) evaluation metric.

A tokenizer-agnostic metric for comparing language models, used by
Karpathy's autoresearch project. Unlike perplexity, BPB is comparable
across models with different tokenizers and vocabulary sizes.

.. math::

    \text{BPB} = \frac{\text{total\_nats}}{\ln(2) \times \text{total\_bytes}}

where total_nats is the sum of cross-entropy losses (in nats, i.e.,
natural log base) over all predicted tokens, and total_bytes is the
total number of UTF-8 bytes in the original text.

Lower BPB is better. For byte-level tokenizers (1 byte per token),
:math:`2^{\text{BPB}}` equals perplexity.

Reference:
    Karpathy, A. (2025). "autoresearch" -- uses val_bpb as the
    primary metric for automated experiment iteration.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as f


def compute_bpb(
    model: nn.Module,
    token_sequences: torch.Tensor,
    bytes_per_token: float,
    device: str = 'cpu',
) -> float:
    r"""Compute bits-per-byte of a language model on token sequences.

    Args:
        model: A language model returning logits of shape
            ``[batch, seq_len, vocab_size]``.
        token_sequences: Token IDs ``[num_sequences, seq_len]``.
        bytes_per_token: Average number of UTF-8 bytes per token.
            For byte-level tokenizers this is 1.0. For BPE tokenizers,
            compute as ``total_bytes / total_tokens`` on your corpus.
        device: Device to run evaluation on.

    Returns:
        BPB as a float. Lower is better.

    Raises:
        ValueError: If ``token_sequences`` is empty.
    """
    if token_sequences.shape[0] == 0:
        raise ValueError('Empty token sequences')

    was_training = model.training
    model.eval()
    device_obj = torch.device(device)
    model.to(device_obj)

    # Get context length from model config
    config = getattr(model, 'config', None)
    if config is not None and hasattr(config, 'context_length'):
        ctx_len: int = config.context_length
    else:
        ctx_len = token_sequences.shape[1] - 1

    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for seq in token_sequences:
            seq_len = seq.shape[0]

            for begin in range(0, seq_len - 1, ctx_len):
                end = min(begin + ctx_len + 1, seq_len)
                inp = seq[begin : end - 1].unsqueeze(0).to(device_obj)
                tgt = seq[begin + 1 : end].to(device_obj)

                logits = model(inp).squeeze(0)

                loss = f.cross_entropy(logits, tgt, reduction='sum')
                total_nll += loss.item()
                total_tokens += tgt.numel()

    # BPB = total_nats / (ln(2) * total_bytes)
    total_bytes = total_tokens * bytes_per_token
    bpb = total_nll / (math.log(2) * total_bytes)

    if was_training:
        model.train()

    return bpb
