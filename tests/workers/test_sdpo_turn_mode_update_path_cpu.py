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
"""End-to-end CPU run of the SDPO turn-mode update path.

Drives the real production chain that unit tests miss: ``attach_response_keep_positions``
-> ``FSDPEngineWithLMHead.forward_backward_batch`` (span-only ``logits_to_keep``,
``distillation_use_topk`` processor contract) -> ``sdpo_ppo_loss`` as both logits
processor and final loss -> the real ``_compute_sdpo_teacher_logps_for_loss`` (teacher
engine forward with spliced rows and the hints-only skip) -> ``loss.backward()``.
Only FSDP wrapping, Ray, and TransferQueue are stubbed."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

import verl.workers.engine.fsdp.transformer_impl as transformer_impl
import verl.workers.utils.losses as sdpo_losses
import verl.workers.utils.padding as padding_mod
from verl.trainer.ppo.sdpo import splice
from verl.trainer.ppo.sdpo.hints import HintedTurn
from verl.trainer.ppo.sdpo.teacher_meta import DEGENERATE_META
from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.utils.losses import sdpo_ppo_loss
from verl.workers.utils.sdpo import attach_response_keep_positions

VOCAB, HIDDEN, TOPK = 64, 8, 5


def njt(rows):
    return torch.nested.nested_tensor(rows, layout=torch.jagged)


@pytest.fixture(scope="module")
def single_process_group(tmp_path_factory):
    if not dist.is_initialized():
        store_file = tmp_path_factory.mktemp("pg") / "store"
        dist.init_process_group(backend="gloo", init_method=f"file://{store_file}", world_size=1, rank=0)
    yield
    dist.destroy_process_group()


@pytest.fixture
def cpu_ops(monkeypatch):
    def unpad_input_cpu(hidden_states, attention_mask, *args, **kwargs):
        mask = attention_mask.bool()
        seqlens = mask.sum(dim=1)
        indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
        flat = hidden_states.reshape(-1, *hidden_states.shape[2:])
        cu_seqlens = torch.nn.functional.pad(seqlens.cumsum(0), (1, 0)).to(torch.int64)
        return flat[indices], indices, cu_seqlens, int(seqlens.max())

    monkeypatch.setattr(padding_mod, "unpad_input", unpad_input_cpu)
    monkeypatch.setattr(padding_mod, "index_first_axis", lambda t, idx: t.index_select(0, idx))
    monkeypatch.setattr(sdpo_losses, "index_first_axis", lambda t, idx: t.index_select(0, idx))
    monkeypatch.setattr(sdpo_losses, "rearrange", lambda t, pattern: t.reshape(-1, t.shape[-1]))
    monkeypatch.setattr(sdpo_losses, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(transformer_impl, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(transformer_impl, "get_device_id", lambda: "cpu")
    import verl.workers.engine.base as engine_base

    monkeypatch.setattr(engine_base, "get_device_name", lambda: "cpu")


class ToyLM(nn.Module):
    """HF CausalLM stand-in: embedding + lm_head honoring tensor ``logits_to_keep``."""

    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(64, HIDDEN)
        self.head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, logits_to_keep=None, **kwargs):
        hidden = self.emb(input_ids)  # (1, nnz, H)
        if logits_to_keep is not None and isinstance(logits_to_keep, torch.Tensor):
            hidden = hidden[:, logits_to_keep, :]
        return SimpleNamespace(logits=self.head(hidden))


def make_engine(module):
    eng = object.__new__(FSDPEngineWithLMHead)
    eng.module = module
    eng.engine_config = SimpleNamespace(
        ulysses_sequence_parallel_size=1,
        fsdp_size=1,
        forward_only=True,
        entropy_from_logits_with_chunking=False,
        use_torch_compile=False,
        entropy_checkpointing=False,
    )
    eng.model_config = SimpleNamespace(use_fused_kernels=False, get=lambda key, default=None: default)
    eng.ulysses_device_mesh = None
    eng.ulysses_parallel_group = None
    eng.ulysses_sequence_parallel_size = 1
    eng.use_ulysses_sp = False
    eng.use_remove_padding = True
    eng._is_offload_param = False
    eng._is_offload_optimizer = False
    eng._autocast_dtype = torch.float32
    eng.scaler = None
    eng._inference_module = None
    eng.mode = None
    return eng


def make_batch(include_hinted=True, include_unhinted=True):
    """Build the turn-mode mini-batch exactly as _maybe_build_self_distillation_batch ships it."""
    rows = []
    if include_hinted:
        # hinted sample: prompt 4, response 12, one hinted turn covering response [2, 7)
        prompt = torch.arange(4, dtype=torch.long) + 1
        resp = torch.arange(12, dtype=torch.long) + 5
        hinted = [HintedTurn(1, 2, 7, "h1", "turn")]
        hint_ids = [torch.tensor([50, 51, 52], dtype=torch.long)]
        header = torch.tensor([60, 61], dtype=torch.long)
        seq, meta, _, spans = splice.build_spliced_teacher_row(prompt, resp, hinted, hint_ids, 4096, header)
        sd_mask = splice.turn_token_mask(12, spans)
        rows.append(dict(prompt=prompt, resp=resp, teacher_seq=seq, meta=torch.tensor(meta), sd_mask=sd_mask))
    if include_unhinted:
        # un-hinted: degenerate 1-token teacher row, zero mask (trainer's else-branch)
        prompt = torch.arange(3, dtype=torch.long) + 1
        resp = torch.arange(9, dtype=torch.long) + 20
        rows.append(
            dict(
                prompt=prompt,
                resp=resp,
                teacher_seq=torch.cat([prompt[-1:], resp[:1]]),
                meta=torch.tensor(DEGENERATE_META, dtype=torch.int64),
                sd_mask=torch.zeros(9),
            )
        )

    # integer masks, matching what the rollout ships through TransferQueue
    response_mask = [torch.ones(r["resp"].shape[0], dtype=torch.long) for r in rows]
    # zero a token to mimic tool-observation masking
    response_mask[0][0] = 0
    # mirrors the trainer: response_mask * mask_row cast to the response_mask dtype
    loss_mask = [response_mask[i] * rows[i]["sd_mask"].to(response_mask[i].dtype) for i in range(len(rows))]

    tensor_dict = {
        "input_ids": njt([torch.cat([r["prompt"], r["resp"]]) for r in rows]),
        "position_ids": njt([torch.arange(r["prompt"].shape[0] + r["resp"].shape[0]) for r in rows]),
        "prompts": njt([r["prompt"] for r in rows]),
        "responses": njt([r["resp"] for r in rows]),
        "response_mask": njt(response_mask),
        "loss_mask": njt(loss_mask),
        "old_log_probs": njt([torch.randn(r["resp"].shape[0]) * 0.1 for r in rows]),
        "rollout_is_weights": njt([torch.ones(r["resp"].shape[0]) for r in rows]),
        "teacher_input_ids": njt([r["teacher_seq"] for r in rows]),
        "teacher_seq_meta": njt([r["meta"] for r in rows]),
        "self_distillation_mask": njt([r["sd_mask"] for r in rows]),
    }
    non_tensor_dict = {
        "use_remove_padding": True,
        "use_fused_kernels": False,
        "use_dynamic_bsz": False,
        "micro_batch_size_per_gpu": 1,
        "calculate_entropy": False,
        "distillation_use_topk": True,
        "temperature": 1.0,
        "pad_token_id": 0,
        "global_batch_size": len(rows),
    }
    return tu.get_tensordict(tensor_dict=tensor_dict, non_tensor_dict=non_tensor_dict)


def run_update(data):
    """Mirror ActorRolloutRefWorker.update_actor minus Ray/FSDP/TQ."""
    student_engine = make_engine(ToyLM(seed=0))
    student_engine.engine_config.forward_only = False
    teacher_engine = make_engine(ToyLM(seed=1))

    worker = SimpleNamespace(
        ref=SimpleNamespace(
            engine=teacher_engine,
            model_config={"use_remove_padding": True},
            engine_config=SimpleNamespace(use_fused_kernels=False),
        )
    )
    teacher_fn = partial(ActorRolloutRefWorker._compute_sdpo_teacher_logps_for_loss, worker)

    actor_config = SimpleNamespace(
        global_batch_info={}, loss_scale_factor=None, loss_agg_mode="token-mean", entropy_coeff=0.0
    )
    sdpo_config = SimpleNamespace(
        full_logit_distillation=True,
        distillation_topk=TOPK,
        distillation_add_tail=True,
        alpha=0.5,
        is_clip=2.0,
    )
    loss_fn = partial(sdpo_ppo_loss, config=actor_config, sdpo_config=sdpo_config, teacher_logprob_fn=teacher_fn)

    attach_response_keep_positions(data)
    outputs = student_engine.forward_backward_batch(data, loss_fn, forward_only=False)
    return student_engine, outputs


def test_turn_mode_update_path_backprops(single_process_group, cpu_ops):
    data = make_batch(include_hinted=True, include_unhinted=True)
    student_engine, outputs = run_update(data)

    metric = outputs["metrics"]["actor/pg_loss"]
    for m in metric if isinstance(metric, list) else [metric]:
        val = m.aggregate() if hasattr(m, "aggregate") else m
        val = val if isinstance(val, torch.Tensor) else torch.tensor(float(val))
        assert torch.isfinite(val).all()

    grads = [p.grad for p in student_engine.module.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0, "hinted spans must produce gradient"


def test_all_unhinted_micro_batch_is_a_finite_noop(single_process_group, cpu_ops):
    data = make_batch(include_hinted=False, include_unhinted=True)
    student_engine, _ = run_update(data)

    grads = [p.grad for p in student_engine.module.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) == 0, "un-hinted rows must contribute zero gradient"


def test_teacher_scores_only_hinted_rows(single_process_group, cpu_ops):
    data = make_batch(include_hinted=True, include_unhinted=True)
    teacher_engine = make_engine(ToyLM(seed=1))
    worker = SimpleNamespace(
        ref=SimpleNamespace(
            engine=teacher_engine,
            model_config={"use_remove_padding": True},
            engine_config=SimpleNamespace(use_fused_kernels=False),
        )
    )
    tu.assign_non_tensor(data, dp_size=1)
    topk_indices = torch.zeros(2, 12, TOPK, dtype=torch.long)
    result = ActorRolloutRefWorker._compute_sdpo_teacher_logps_for_loss(
        worker, data=data, student_topk_indices=topk_indices, return_all_logps=False
    )

    teacher_lp = result["teacher_log_probs"]
    assert teacher_lp.shape == (2, 12)
    assert teacher_lp[0, 2:7].abs().sum() > 0, "hinted span must be scored"
    assert teacher_lp[0, :2].abs().sum() == 0 and teacher_lp[0, 7:].abs().sum() == 0
    # un-hinted row: only the degenerate (loss-masked) first position is scored
    assert teacher_lp[1, 0].abs() > 0 and teacher_lp[1, 1:].abs().sum() == 0
    assert result["teacher_topk_log_probs"].shape == (2, 12, TOPK)


def test_all_unhinted_micro_still_runs_teacher(single_process_group, cpu_ops):
    """dp-collective lockstep: the teacher forward runs even when no row is hinted."""
    data = make_batch(include_hinted=False, include_unhinted=True)
    teacher_engine = make_engine(ToyLM(seed=1))
    worker = SimpleNamespace(
        ref=SimpleNamespace(
            engine=teacher_engine,
            model_config={"use_remove_padding": True},
            engine_config=SimpleNamespace(use_fused_kernels=False),
        )
    )
    tu.assign_non_tensor(data, dp_size=1)
    topk_indices = torch.zeros(1, 9, TOPK, dtype=torch.long)
    result = ActorRolloutRefWorker._compute_sdpo_teacher_logps_for_loss(
        worker, data=data, student_topk_indices=topk_indices, return_all_logps=False
    )
    teacher_lp = result["teacher_log_probs"]
    assert teacher_lp[0, 0].abs() > 0, "teacher must score the degenerate token (lockstep)"
    assert teacher_lp[0, 1:].abs().sum() == 0
    assert result["teacher_topk_log_probs"].shape == (1, 9, TOPK)
