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

"""Implementation of various attention mechanisms."""

from .flash_attention import FlashAttention
from .grouped_query_attention import GroupedQueryAttention
from .kv_cache import KVCache
from .multi_head_latent_attention import MultiHeadLatentAttention
from .multihead_attention import MultiHeadAttention
from .sliding_window_attention import SlidingWindowAttention

ATTENTION_REGISTRY: dict[str, type] = {
    'mha': MultiHeadAttention,
    'gqa': GroupedQueryAttention,
    'sliding_window': SlidingWindowAttention,
    'mla': MultiHeadLatentAttention,
    'flash': FlashAttention,
}

__all__ = [
    'MultiHeadAttention',
    'FlashAttention',
    'GroupedQueryAttention',
    'SlidingWindowAttention',
    'MultiHeadLatentAttention',
    'KVCache',
    'ATTENTION_REGISTRY',
]
