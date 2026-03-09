"""BPE tokenizer trainer for custom corpora.

Trains a Byte-Pair Encoding tokenizer from scratch on a given text
corpus using the HuggingFace ``tokenizers`` library (fast Rust
implementation). Returns an LMT-compatible tokenizer ready for
training pipelines.

Useful for BabyLM Challenge, domain-specific pretraining, and
any scenario where a pretrained tokenizer doesn't match the data.

Usage::

    from lmt.tokenizer.trainer import train_bpe_tokenizer

    texts = ['The quick brown fox...', 'Another sentence...']
    tok = train_bpe_tokenizer(texts, vocab_size=8192)

    ids = tok.encode('Hello world')
    text = tok.decode(ids)

    # Save for later use
    tok.save('my_tokenizer')

    # Load it back
    from lmt.tokenizer.trainer import load_trained_tokenizer

    tok = load_trained_tokenizer('my_tokenizer')

Requires the ``tokenizers`` package (installed with ``transformers``).
"""

from __future__ import annotations

from pathlib import Path

from lmt.tokenizer.hf_tokenizer import HFTokenizerWrapper


def train_bpe_tokenizer(
    texts: list[str],
    vocab_size: int = 8192,
    min_frequency: int = 2,
    special_tokens: list[str] | None = None,
) -> HFTokenizerWrapper:
    """Train a BPE tokenizer from scratch on a text corpus.

    Uses the HuggingFace ``tokenizers`` library for fast training.
    Returns an ``HFTokenizerWrapper`` compatible with all LMT
    training pipelines.

    Args:
        texts: List of text strings to train on.
        vocab_size: Target vocabulary size.
        min_frequency: Minimum frequency for a pair to be merged.
            Higher values = fewer merges = simpler tokenizer.
        special_tokens: Optional list of special token strings
            (e.g. ``['<pad>', '<eos>']``). Added to vocabulary
            before BPE training.

    Returns:
        Trained tokenizer wrapped in ``HFTokenizerWrapper``.

    Raises:
        ValueError: If ``texts`` is empty.
        ImportError: If ``tokenizers`` is not installed.
    """
    if not texts:
        raise ValueError('Cannot train tokenizer on empty corpus.')

    try:
        from tokenizers import Tokenizer  # type: ignore[import-untyped]
        from tokenizers.decoders import (  # type: ignore[import-untyped]
            ByteLevel as ByteLevelDecoder,
        )
        from tokenizers.models import BPE  # type: ignore[import-untyped]
        from tokenizers.pre_tokenizers import (  # type: ignore[import-untyped]
            ByteLevel,
        )
        from tokenizers.trainers import (  # type: ignore[import-untyped]
            BpeTrainer,
        )
    except ImportError as e:
        raise ImportError(
            'The `tokenizers` package is required for tokenizer '
            'training. Install with: pip install tokenizers'
        ) from e

    if special_tokens is None:
        special_tokens = []

    # Build tokenizer with byte-level BPE (like GPT-2)
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # Train on the corpus
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Wrap in HuggingFace PreTrainedTokenizerFast for compatibility
    try:
        from transformers import (  # type: ignore[import-untyped]
            PreTrainedTokenizerFast,
        )
    except ImportError as e:
        raise ImportError(
            'The `transformers` package is required to wrap the '
            'trained tokenizer. Install with: pip install transformers'
        ) from e

    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

    wrapper = HFTokenizerWrapper(hf_tokenizer)
    # Attach save method for convenience
    wrapper.save = lambda path: _save_tokenizer(hf_tokenizer, path)  # type: ignore[attr-defined]
    return wrapper


def _save_tokenizer(
    hf_tokenizer: object,
    path: str,
) -> None:
    """Save a trained tokenizer to disk.

    Args:
        hf_tokenizer: HuggingFace tokenizer to save.
        path: Directory path to save to.
    """
    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)
    hf_tokenizer.save_pretrained(str(save_dir))  # type: ignore[union-attr]


def load_trained_tokenizer(path: str) -> HFTokenizerWrapper:
    """Load a previously trained tokenizer from disk.

    Args:
        path: Directory path where tokenizer was saved.

    Returns:
        Loaded tokenizer wrapped in ``HFTokenizerWrapper``.
    """
    try:
        from transformers import (  # type: ignore[import-untyped]
            PreTrainedTokenizerFast,
        )
    except ImportError as e:
        raise ImportError(
            'The `transformers` package is required. '
            'Install with: pip install transformers'
        ) from e

    hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(path)
    wrapper = HFTokenizerWrapper(hf_tokenizer)
    wrapper.save = lambda p: _save_tokenizer(hf_tokenizer, p)  # type: ignore[attr-defined]
    return wrapper
