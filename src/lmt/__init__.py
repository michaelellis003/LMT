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

"""A Python package for language modeling using Transformers.

For educational and research purposes. This Python is to help users learn
how the attention mechanism and Transformer architecture (Vaswani et al., 2017)
in their various forms are used for natural language modeling and in
particular their use in Large Language Models (LLM).
"""

from .models import ModelConfig
from .models.gemma import Gemma
from .models.gpt import GPT
from .models.llama import LLaMA
from .models.mamba import Mamba
from .models.qwen3 import Qwen3
from .tokenizer import (
    BaseTokenizer,
    BPETokenizer,
    CharTokenizer,
    NaiveTokenizer,
)
from .training import Trainer

__all__ = [
    'ModelConfig',
    'Gemma',
    'GPT',
    'LLaMA',
    'Mamba',
    'Qwen3',
    'Trainer',
    'BaseTokenizer',
    'BPETokenizer',
    'CharTokenizer',
    'NaiveTokenizer',
]
