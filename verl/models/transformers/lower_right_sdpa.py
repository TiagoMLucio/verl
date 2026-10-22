# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""SDPA for forwards that continue a KV cache. transformers gives such a forward (fewer queries than keys) a dense
(query x key) mask, which keeps SDPA off its flash kernel; here a causal mask over a row without padding is left out
and the attention applies PyTorch's lower-right causal bias instead, which the flash and memory-efficient kernels run
without a mask. Anything else falls back to transformers' SDPA."""

from contextlib import contextmanager

import torch

NAME = "sdpa_lower_right"
_registered = False


def _mask(*args, attention_mask=None, kv_length=None, kv_offset=0, mask_function=None, **kwargs):
    from transformers.masking_utils import causal_mask_function, prepare_padding_mask, sdpa_mask

    padding = prepare_padding_mask(attention_mask, kv_length, kv_offset)
    if mask_function is causal_mask_function and (padding is None or bool(padding.all())):
        return None
    return sdpa_mask(
        *args, attention_mask=attention_mask, kv_length=kv_length, kv_offset=kv_offset, mask_function=mask_function,
        **kwargs,
    )


def _attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
    from torch.nn.attention.bias import causal_lower_right
    from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward

    q, k = query.shape[2], key.shape[2]
    if attention_mask is None and 1 < q < k:
        groups = query.shape[1] // key.shape[1]
        if groups > 1:
            # the lower-right bias reaches the flash kernel without enable_gqa
            key, value = repeat_kv(key, groups), repeat_kv(value, groups)
        out = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=causal_lower_right(q, k), dropout_p=0.0, scale=scaling
        )
        return out.transpose(1, 2).contiguous(), None
    return sdpa_attention_forward(
        module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs
    )


def available(model: torch.nn.Module) -> bool:
    """Whether ``model`` is configured for transformers' SDPA and transformers has the attention interfaces."""
    try:
        from transformers import AttentionInterface  # noqa: F401
        from transformers.masking_utils import AttentionMaskInterface  # noqa: F401
    except ImportError:
        return False
    return getattr(getattr(model, "config", None), "_attn_implementation", None) == "sdpa"


@contextmanager
def lower_right_sdpa(model: torch.nn.Module):
    """Inside, a model configured for transformers' SDPA attends through the lower-right path. Yields whether it
    switched: a model on another attention, or a transformers without the interfaces, is left as it is."""
    global _registered
    if not available(model):
        yield False
        return
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface

    if not _registered:
        AttentionInterface.register(NAME, _attention)
        AttentionMaskInterface.register(NAME, _mask)
        _registered = True
    model.config._attn_implementation = NAME
    try:
        yield True
    finally:
        model.config._attn_implementation = "sdpa"
