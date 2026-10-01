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
"""The padded span-only lm_head branch divides the gathered (n_keep x vocab) logits by the
temperature only when that is not a no-op. A unit temperature must return exactly what the
divide by 1 returned; any other temperature, scalar or per row, is still applied."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import verl.workers.engine.fsdp.transformer_impl as transformer_impl
from verl.utils import tensordict_utils as tu
from verl.utils import torch_functional as verl_F
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead, _is_unit_temperature

VOCAB, HIDDEN = 64, 8
ROW_LENGTHS = (9, 6)
KEEP = ([3, 4, 5, 6], [1, 2, 3])


def njt(rows):
    return torch.nested.nested_tensor(rows, layout=torch.jagged)


class ToyLM(nn.Module):
    """HF CausalLM stand-in on the padded layout: tensor ``logits_to_keep`` picks the same
    sequence positions on every row, as HF does."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(VOCAB, HIDDEN)
        self.head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, input_ids=None, logits_to_keep=None, **kwargs):
        hidden = self.emb(input_ids)
        if logits_to_keep is not None:
            hidden = hidden[:, logits_to_keep, :]
        return SimpleNamespace(logits=self.head(hidden))


def make_engine():
    eng = object.__new__(FSDPEngineWithLMHead)
    eng.module = ToyLM()
    eng.engine_config = SimpleNamespace(entropy_checkpointing=False)
    eng.model_config = SimpleNamespace(use_fused_kernels=False, get=lambda key, default=None: default)
    eng.use_ulysses_sp = False
    eng.compute_entropy_from_logits = verl_F.entropy_from_logits
    return eng


def make_micro_batch(temperature):
    torch.manual_seed(1)
    rows = [torch.randint(1, VOCAB, (n,)) for n in ROW_LENGTHS]
    tensor_dict = {
        "input_ids": njt(rows),
        "position_ids": njt([torch.arange(n) for n in ROW_LENGTHS]),
        "logits_keep_positions": njt([torch.tensor(k) for k in KEEP]),
    }
    non_tensor_dict = {
        "use_remove_padding": False,
        "use_fused_kernels": False,
        "calculate_entropy": True,
        "distillation_use_topk": True,
        "pad_token_id": 0,
    }
    if isinstance(temperature, torch.Tensor):
        tensor_dict["temperature"] = temperature
    else:
        non_tensor_dict["temperature"] = temperature
    return rows, tu.get_tensordict(tensor_dict=tensor_dict, non_tensor_dict=non_tensor_dict)


def span_only_forward(temperature):
    """prepare_model_inputs -> padded forward -> prepare_model_outputs, keeping the logits the
    distillation processor was handed."""
    engine = make_engine()
    rows, micro_batch = make_micro_batch(temperature)
    model_inputs, output_args = engine.prepare_model_inputs(micro_batch)
    assert "keep_union" in output_args

    seen = {}

    def processor(student_logits, data, logits_keep_idx):
        seen["logits"] = student_logits
        return {}

    model_output = engine.prepare_model_outputs(engine.module(**model_inputs), output_args, micro_batch, processor)
    return rows, engine, output_args, model_output, seen["logits"]


def fed_from(student_logits):
    """The op that produced the processor's (n_keep, vocab) logits, under the unsqueeze(0)."""
    return type(student_logits.grad_fn.next_functions[0][0]).__name__


@pytest.mark.parametrize("temperature", [1, 1.0, torch.ones(2), torch.ones(2, dtype=torch.bfloat16)])
def test_unit_temperature_is_detected(temperature):
    assert _is_unit_temperature(temperature) is True


@pytest.mark.parametrize("temperature", [0.7, 2, torch.tensor([1.0, 0.5]), torch.full((2,), 0.7)])
def test_other_temperatures_are_not(temperature):
    assert _is_unit_temperature(temperature) is False


@pytest.mark.parametrize("temperature", [1.0, torch.ones(2)], ids=["scalar", "per-row"])
def test_unit_temperature_skips_the_divide_and_changes_nothing(temperature, monkeypatch):
    _, _, output_args, skipped, logits = span_only_forward(temperature)
    assert output_args["temperature_is_one"] is True
    assert fed_from(logits) == "CatBackward0"

    monkeypatch.setattr(transformer_impl, "_is_unit_temperature", lambda t: False)
    _, _, _, divided, divided_logits = span_only_forward(temperature)
    assert fed_from(divided_logits) == "DivBackward0"

    torch.testing.assert_close(logits, divided_logits, rtol=0, atol=0)
    for key in ("log_probs", "entropy"):
        torch.testing.assert_close(skipped[key].values(), divided[key].values(), rtol=0, atol=0)


@pytest.mark.parametrize("temperature", [0.7, torch.tensor([1.0, 0.5])], ids=["scalar", "per-row"])
def test_other_temperatures_are_applied_per_row(temperature):
    rows, engine, output_args, model_output, logits = span_only_forward(temperature)
    assert output_args["temperature_is_one"] is False
    assert fed_from(logits) == "DivBackward0"

    per_row = temperature if isinstance(temperature, torch.Tensor) else torch.full((len(rows),), temperature)
    for row, keep, t, log_probs in zip(rows, KEEP, per_row, model_output["log_probs"].unbind(), strict=True):
        keep = torch.tensor(keep)
        expected = torch.log_softmax(engine.module(input_ids=row[None]).logits[0, keep] / t, dim=-1)
        expected = expected.gather(-1, row[keep + 1].unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(log_probs[keep], expected)
