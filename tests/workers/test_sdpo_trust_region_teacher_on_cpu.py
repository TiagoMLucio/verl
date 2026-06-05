# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.workers.config.actor import SelfDistillationConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.utils import losses as sdpo_losses
from verl.workers.utils.losses import _sdpo_logits_processor, _sdpo_teacher_extractor
from verl.workers.utils.padding import no_padding_2_padding
from verl.workers.utils.sdpo import TrustRegionTeacher, reconstruct_padded_teacher_from_nested


class ConstantLogitsModule(torch.nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, *args, **kwargs):
        return SimpleNamespace(logits=self.logits)


class FakeEngine:
    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.inference_module = None
        self.infer_result = {"model_output": {}}

    def eval_mode(self, **kwargs):
        return nullcontext()

    def set_inference_module(self, module: torch.nn.Module):
        self.inference_module = module

    def infer_batch(self, data, loss_function=None):
        self.last_infer_data = data
        self.last_loss_function = loss_function
        return self.infer_result


class FakeTrainingWorker:
    def __init__(self, module: torch.nn.Module, *, use_fused_kernels: bool = False):
        self.engine = FakeEngine(module)
        self.engine_config = SimpleNamespace(use_fused_kernels=use_fused_kernels)


def make_worker(regularization: str, *, use_fused_kernels: bool = False):
    ref_module = ConstantLogitsModule(torch.tensor([[[1.0, 3.0]]]))
    actor_module = ConstantLogitsModule(torch.tensor([[[5.0, 7.0]]]))
    worker = object.__new__(ActorRolloutRefWorker)
    worker.sdpo_enabled = True
    worker._is_ref = True
    worker.sdpo_config = SelfDistillationConfig(
        teacher_regularization=regularization,
        teacher_update_rate=0.25,
        ema_update_rate=0.05,
    )
    worker.ref = FakeTrainingWorker(ref_module, use_fused_kernels=use_fused_kernels)
    worker.actor = FakeTrainingWorker(actor_module, use_fused_kernels=use_fused_kernels)
    return worker, ref_module, actor_module


def make_padding_data(extra_tensors: dict[str, torch.Tensor]) -> TensorDict:
    input_ids = torch.tensor([[10, 11, 12, 13]])
    response_mask = torch.ones(1, 2, dtype=torch.long)
    data = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "position_ids": torch.arange(4).unsqueeze(0),
            "responses": input_ids[:, -2:],
            "response_mask": response_mask,
            "prompts": input_ids[:, :2],
            **extra_tensors,
        },
        batch_size=[1],
    )
    data.set_non_tensor("max_seq_len", input_ids.shape[1])
    data.set_non_tensor("max_response_len", response_mask.shape[1])
    data.set_non_tensor("indices", torch.arange(input_ids.numel()))
    return data


@pytest.fixture
def cpu_attention_ops(monkeypatch):
    def index_first_axis_cpu(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return tensor.index_select(0, indices)

    def rearrange_cpu(tensor: torch.Tensor, pattern: str) -> torch.Tensor:
        assert pattern == "b s k -> (b s) k"
        return tensor.reshape(-1, tensor.shape[-1])

    monkeypatch.setattr(sdpo_losses, "index_first_axis", index_first_axis_cpu)
    monkeypatch.setattr(sdpo_losses, "rearrange", rearrange_cpu)
    monkeypatch.setattr(sdpo_losses, "get_device_name", lambda: "cpu")


def test_trust_region_teacher_blends_ref_and_student_logits():
    ref_logits = torch.tensor([[[0.0, 2.0, 4.0]]])
    student_logits = torch.tensor([[[10.0, 20.0, 30.0]]])
    teacher = TrustRegionTeacher(
        ref_module=ConstantLogitsModule(ref_logits),
        student_module=ConstantLogitsModule(student_logits),
        mix_coef=0.25,
    )

    output = teacher(input_ids=torch.tensor([[1, 2, 3]]))

    torch.testing.assert_close(output.logits, torch.lerp(ref_logits, student_logits, 0.25))


def test_reconstruct_padded_teacher_matches_legacy_layout():
    # Transfer-queue path stores the teacher sequence (teacher prompt + response), responses and
    # response mask as nested per-sample tensors; the worker rebuilds the left-padded prompt +
    # right-padded response layout that the legacy trainer produced. Prompts here are [10,11] and
    # [20,21,22]; the teacher sequence is the prompt concatenated with the response.
    teacher_input_ids_nested = torch.nested.nested_tensor(
        [torch.tensor([10, 11, 1, 2, 3]), torch.tensor([20, 21, 22, 4, 5])], layout=torch.jagged
    )
    responses = torch.nested.nested_tensor(
        [torch.tensor([1, 2, 3]), torch.tensor([4, 5])], layout=torch.jagged
    )
    response_mask = torch.nested.nested_tensor(
        [torch.tensor([1, 1, 1]), torch.tensor([1, 1])], layout=torch.jagged
    )

    teacher_input_ids, teacher_attention_mask, teacher_position_ids, responses_padded, response_mask_padded = (
        reconstruct_padded_teacher_from_nested(teacher_input_ids_nested, responses, response_mask, pad_token_id=0)
    )

    # max_prompt_len=3, max_response_len=3 -> teacher seq len 6
    assert teacher_input_ids.tolist() == [[0, 10, 11, 1, 2, 3], [20, 21, 22, 4, 5, 0]]
    assert teacher_attention_mask.tolist() == [[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]]
    # position ids = clip(cumsum(mask) - 1, min=0)
    assert teacher_position_ids.tolist() == [[0, 0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 4]]
    assert responses_padded.tolist() == [[1, 2, 3], [4, 5, 0]]
    assert response_mask_padded.tolist() == [[1, 1, 1], [1, 1, 0]]
    # the prompt boundary is uniform so the legacy slicing teacher_input_ids[:, :Lp] is valid
    assert teacher_input_ids.shape[1] - responses_padded.shape[1] == 3


@pytest.mark.parametrize("mix_coef", [-0.1, 1.1])
def test_trust_region_teacher_rejects_invalid_mix_coef(mix_coef):
    module = ConstantLogitsModule(torch.zeros(1, 1, 2))

    with pytest.raises(ValueError, match="mix_coef must be in"):
        TrustRegionTeacher(ref_module=module, student_module=module, mix_coef=mix_coef)


@pytest.mark.parametrize(
    ("input_mode", "expected_mode"),
    [
        ("ema", "ema"),
        ("trust_region", "trust_region"),
        ("trust-region", "trust_region"),
        ("trustregion", "trust_region"),
        ("none", "none"),
    ],
)
def test_self_distillation_config_canonicalizes_teacher_regularization(input_mode, expected_mode):
    cfg = SelfDistillationConfig(teacher_regularization=input_mode)

    assert cfg.teacher_regularization == expected_mode


@pytest.mark.parametrize("regularization", ["ema", "none"])
def test_sdpo_teacher_does_not_register_ref_backed_modes(regularization):
    worker, _, _ = make_worker(regularization)

    worker._configure_sdpo_teacher()

    assert worker.ref.engine.inference_module is None


def test_sdpo_teacher_uses_trust_region_module_for_trust_region():
    worker, ref_module, actor_module = make_worker("trust-region")

    worker._configure_sdpo_teacher()

    teacher_module = worker.ref.engine.inference_module
    assert isinstance(teacher_module, TrustRegionTeacher)
    assert teacher_module.ref_module is ref_module
    assert teacher_module.student_module is actor_module
    assert teacher_module.mix_coef == 0.25


def test_sdpo_trust_region_rejects_fused_kernels():
    worker, _, _ = make_worker("trust_region", use_fused_kernels=True)

    with pytest.raises(ValueError, match="trust_region teacher requires disabling fused kernels"):
        worker._configure_sdpo_teacher()


def test_fsdp_engine_uses_single_inference_module_for_forward_only_steps():
    engine = object.__new__(FSDPEngine)
    base_module = ConstantLogitsModule(torch.zeros(1, 1, 2))
    module = ConstantLogitsModule(torch.zeros(1, 1, 2))
    engine.module = base_module
    engine._inference_module = None

    def selected_module(forward_only: bool):
        return engine._inference_module if forward_only and engine._inference_module is not None else engine.module

    assert selected_module(forward_only=True) is base_module
    assert selected_module(forward_only=False) is base_module

    engine.set_inference_module(module)

    assert selected_module(forward_only=True) is module
    assert selected_module(forward_only=False) is base_module


def test_sdpo_teacher_topk_extractor_supports_logits_processor_and_eval_loss_calls(cpu_attention_ops):
    data = make_padding_data(
        {
            "student_topk_indices": torch.tensor([[[2, 0], [1, 3]]]),
        }
    )
    logits = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 1.0, 0.0, 2.0],
            [1.0, 5.0, 2.0, 0.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    ).unsqueeze(0)

    outputs = _sdpo_teacher_extractor(student_logits=logits, data=data)
    loss, metrics = _sdpo_teacher_extractor(model_output={"log_probs": torch.zeros(1)})
    padded_topk_logps = no_padding_2_padding(outputs["topk_logps"], data)
    expected_token0 = torch.log_softmax(logits[0, 1], dim=-1)[torch.tensor([2, 0])]
    expected_token1 = torch.log_softmax(logits[0, 2], dim=-1)[torch.tensor([1, 3])]

    torch.testing.assert_close(padded_topk_logps[0, 0], expected_token0)
    torch.testing.assert_close(padded_topk_logps[0, 1], expected_token1)
    assert loss.shape == torch.Size([])
    assert metrics == {}


def test_sdpo_student_topk_is_computed_from_student_logits():
    data = make_padding_data(
        {
            "teacher_topk_indices": torch.tensor([[[2, 0], [1, 3]]]),
        }
    )
    cfg = SimpleNamespace(full_logit_distillation=True, distillation_topk=2)
    logits = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 1.0, 0.0, 2.0],
            [1.0, 5.0, 2.0, 0.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    ).unsqueeze(0)

    outputs = _sdpo_logits_processor(student_logits=logits, sdpo_config=cfg)
    padded_topk_logps = no_padding_2_padding(outputs["topk_logps"], data)
    padded_topk_indices = no_padding_2_padding(outputs["topk_indices"], data)
    expected_token0_indices = torch.tensor([0, 3])
    expected_token1_indices = torch.tensor([1, 2])
    expected_token0 = torch.log_softmax(logits[0, 1], dim=-1)[expected_token0_indices]
    expected_token1 = torch.log_softmax(logits[0, 2], dim=-1)[expected_token1_indices]

    torch.testing.assert_close(padded_topk_indices[0, 0], expected_token0_indices)
    torch.testing.assert_close(padded_topk_indices[0, 1], expected_token1_indices)
    torch.testing.assert_close(padded_topk_logps[0, 0], expected_token0)
    torch.testing.assert_close(padded_topk_logps[0, 1], expected_token1)


def test_sdpo_teacher_all_logps_extractor_supports_logits_processor_and_eval_loss_calls(cpu_attention_ops):
    logits = torch.tensor([[[1.0, 3.0, 2.0]]])

    # No student_topk_indices in data → extractor returns full-vocab logps
    empty_data = TensorDict({}, batch_size=[1])
    outputs = _sdpo_teacher_extractor(student_logits=logits, data=empty_data)
    loss, metrics = _sdpo_teacher_extractor(model_output={"log_probs": torch.zeros(1)})

    torch.testing.assert_close(outputs["all_logps"].exp().sum(dim=-1), torch.ones(1))
    assert loss.shape == torch.Size([])
    assert metrics == {}
