# Copyright 2025 Michael Ellis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions needed for generating text."""

import torch
import torch.nn as nn

from lmt.tokenizer import BaseTokenizer


def generate(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """Generates a sequence of token IDs based on a prompt.

    This function iteratively predicts the next token in a sequence using the
    provided model. It supports different decoding strategies, including greedy
    decoding, temperature-based sampling, and top-k sampling. The generation
    process can be stopped prematurely if an end-of-sequence (EOS) token is
    produced.

    Args:
        model (nn.Module): The transformer model to use for generation.
        idx (torch.Tensor): The initial sequence of token IDs, representing the
            prompt. Shape: (batch_size, sequence_length).
        max_new_tokens (int): The maximum number of new tokens to generate.
        context_size (int): The model's context window size. The
            input sequence will be cropped from the left to this size if it is
            too long.
        temperature (float, optional): Controls the randomness of the
            predictions. A value of 0.0 corresponds to greedy decoding (always
            picking the most likely token). Higher values increase randomness.
            Defaults to 1.0.
        top_k (int | None, optional): If set, the model's predictions are
            filtered to the `k` most likely next tokens before sampling.
            This can help reduce the chance of generating low-probability
            tokens. Defaults to None.
        eos_id (int | None, optional): The token ID that signifies the end of a
            sequence. If this token is generated across all batches, the
            function will stop generating new tokens. Defaults to None.

    Returns:
        torch.Tensor: The generated sequence of token IDs, which includes the
            original prompt plus the newly generated tokens. The length of the
            newly generated part is at most `max_new_tokens`. Shape:
            (batch_size, sequence_length + num_generated_tokens).
    """
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1].unsqueeze(-1)  # [batch, 1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits,
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # Get the idx of the vocab entry with the highest logits value
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        if eos_id is not None and (idx_next == eos_id).all():
            break

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def generate_and_print_sample(
    model: nn.Module,
    tokenizer: BaseTokenizer,
    device: torch.device,
    start_context: str,
):
    """Generates and prints a sample of text from the model.

    This function sets the model to evaluation mode, generates a text sample
    based on a starting context, prints the result to the console, and then
    restores the model to training mode.

    Works with any model that has a ``config.context_length`` attribute
    (BaseModel, GPT, LLaMA, Mamba, etc.) or a ``pos_embed`` attribute
    (legacy GPT-style).

    Args:
        model (nn.Module): The transformer model to use for generation.
        tokenizer (Any): The tokenizer used for encoding the start context
            and decoding the generated token IDs. The specific type can vary.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') on which
            to perform the generation.
        start_context (str): The initial string to prompt the model.
    """
    model.eval()
    config = getattr(model, 'config', None)
    if config is not None and hasattr(config, 'context_length'):
        context_size: int = config.context_length
    else:
        context_size = int(model.pos_embed.weight.shape[0])  # type: ignore

    encoded = tokenizer.encode(start_context)
    encoded = torch.tensor(encoded).unsqueeze(0).to(device)

    with torch.no_grad():
        token_ids = generate(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
        )

        flat = token_ids.squeeze(0)  # remove batch dimension
        decoded_text = tokenizer.decode(flat.tolist())
        print(decoded_text.replace('\n', ' '))  # Compact print format
    model.train()
