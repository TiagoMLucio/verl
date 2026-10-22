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

"""The lower-right SDPA path for forwards continuing a cache: same attention as a dense causal mask, no mask built
for an unpadded causal row, and the model's attention switched only inside the context."""

from types import SimpleNamespace

import torch
from transformers import PretrainedConfig
from transformers.masking_utils import causal_mask_function

from verl.models.transformers import lower_right_sdpa as lr


def test_continuing_queries_attend_as_under_a_dense_causal_mask():
    torch.manual_seed(0)
    heads, kv_heads, q, k, dim = 8, 2, 5, 12, 16
    query = torch.randn(1, heads, q, dim)
    key, value = torch.randn(1, kv_heads, k, dim), torch.randn(1, kv_heads, k, dim)
    module = SimpleNamespace(is_causal=True, num_key_value_groups=heads // kv_heads, training=False)
    out, _ = lr._attention(module, query, key, value, None, scaling=dim**-0.5)
    # the last q of k positions: query i sees the keys up to k - q + i
    mask = torch.ones(q, k, dtype=torch.bool).tril(diagonal=k - q)
    rep = heads // kv_heads
    expected = torch.nn.functional.scaled_dot_product_attention(
        query, key.repeat_interleave(rep, 1), value.repeat_interleave(rep, 1), attn_mask=mask, scale=dim**-0.5
    )
    torch.testing.assert_close(out, expected.transpose(1, 2), atol=1e-5, rtol=1e-5)


def test_no_mask_for_an_unpadded_causal_row_and_a_mask_with_padding():
    common = dict(batch_size=1, q_length=5, kv_length=12, q_offset=7, kv_offset=0, mask_function=causal_mask_function)
    assert lr._mask(**common, attention_mask=torch.ones(1, 12, dtype=torch.bool)) is None
    padded = torch.ones(1, 12, dtype=torch.bool)
    padded[0, 0] = False
    mask = lr._mask(**common, attention_mask=padded)
    assert mask is not None and not bool(mask[..., 0].any())


def _model(attn: str) -> torch.nn.Module:
    model = torch.nn.Module()
    model.config = PretrainedConfig()
    model.config._attn_implementation = attn
    return model


def test_switches_an_sdpa_model_inside_and_restores_it():
    model = _model("sdpa")
    with lr.lower_right_sdpa(model) as switched:
        assert switched and model.config._attn_implementation == lr.NAME
    assert model.config._attn_implementation == "sdpa"


def test_leaves_other_attention_alone():
    model = _model("eager")
    assert not lr.available(model)
    with lr.lower_right_sdpa(model) as switched:
        assert not switched and model.config._attn_implementation == "eager"
