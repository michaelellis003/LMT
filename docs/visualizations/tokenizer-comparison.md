# Tokenizer Comparison

How a model "sees" text depends entirely on its tokenizer. The same
sentence can be 40 tokens (character-level) or 8 tokens (BPE), and
this dramatically affects what the model needs to learn.

## Interactive Explorer

<div class="tokenizer-viz"></div>

## Why Tokenization Matters

### Vocabulary Size vs Sequence Length

There's a fundamental trade-off:

| Tokenizer | Vocab Size | Sequence Length | What Model Learns |
|-----------|-----------|-----------------|-------------------|
| Character | ~100 | Very long | Must learn spelling, then words, then meaning |
| Word | ~50,000+ | Short | Already knows words, but can't handle typos or new words |
| BPE (subword) | ~50,000 | Medium | Learns common subwords; handles novel words by composition |

### The BPE Insight

Byte Pair Encoding starts with individual characters and iteratively
merges the most frequent pair. After enough merges:

- Common words become single tokens: `"the"` → `[the]`
- Rare words get split into pieces: `"unhappiness"` → `[un, happi, ness]`
- Unknown words still work: `"transformerize"` → `[transform, er, ize]`

This is why GPT-2, GPT-4, and LLaMA all use BPE variants -- it balances
vocabulary size and sequence length while handling any input text.

### Compression Ratio

The **compression ratio** (characters ÷ tokens) tells you how efficient
a tokenizer is. Higher compression means the model sees more "meaning"
per token:

- Character: 1.0x (no compression)
- BPE on English: ~3-4x (each token ≈ 3-4 characters)
- Word: ~5-6x (but huge vocabulary and no subword flexibility)

## LMT Tokenizers

LMT provides three tokenizers:

```python
from lmt import CharTokenizer, BPETokenizer, NaiveTokenizer

# Character-level: great for small experiments, no dependencies
char_tok = CharTokenizer.from_text("hello world")
char_tok.encode("hello")  # [4, 1, 7, 7, 8]

# BPE: production-quality, uses tiktoken (GPT-2 vocabulary)
bpe_tok = BPETokenizer(encoding_name='gpt2')
bpe_tok.encode("hello world")  # [31373, 995]

# Naive: word-level split for testing
naive_tok = NaiveTokenizer(text="hello world")
```

Choose character-level for quick experiments (no dependencies, tiny vocab),
and BPE for real training (better compression, handles any text).
