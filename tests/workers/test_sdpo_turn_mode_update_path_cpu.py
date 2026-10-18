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

import math
from functools import partial
from itertools import chain
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

import verl.workers.engine.fsdp.transformer_impl as transformer_impl
import verl.workers.utils.losses as sdpo_losses
import verl.workers.utils.padding as padding_mod
from verl.trainer.ppo.core_algos import SDPO_SPAN_ROWS_KEY, finalize_ratio_metrics
from verl.trainer.ppo.sdpo.teacher_meta import DEGENERATE_META, SubRow, pack
from verl.utils import tensordict_utils as tu
from verl.utils.experimental.chunked_topk import chunked_gather_logprobs
from verl.utils.experimental.torch_functional import FusedLinearForPPO
from verl.utils.metric import Metric
from verl.utils.torch_functional import allgather_dict_into_dict
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.utils.losses import sdpo_ppo_loss
from verl.workers.utils.sdpo import (
    _raise_if_partly_on_the_grid,
    attach_response_keep_positions,
    truncate_rows_after_last_supervised_token,
)

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


class ToyChunkedLM(ToyLM):
    """ToyLM with the patched head's teacher mode: it scores the ids it is handed a chunk at a time
    and has no logits to give back."""

    def __init__(self, seed):
        super().__init__(seed)
        self.scored = []

    def forward(
        self,
        input_ids=None,
        logits_to_keep=None,
        distill_indices=None,
        distill_chunk_size=512,
        temperature=1.0,
        **kwargs,
    ):
        if distill_indices is None:
            return super().forward(input_ids=input_ids, logits_to_keep=logits_to_keep, **kwargs)
        kept = self.emb(input_ids)[:, logits_to_keep, :].squeeze(0)
        labels = torch.roll(input_ids, shifts=-1, dims=-1).squeeze(0)[logits_to_keep]
        logps, label_logps = chunked_gather_logprobs(
            kept, self.head.weight, distill_indices, labels, temperature, distill_chunk_size
        )
        self.scored.append((tuple(distill_indices.shape), distill_chunk_size, temperature))
        return SimpleNamespace(log_probs=label_logps.unsqueeze(0), topk_logps=logps.unsqueeze(0), topk_indices=None)


class ToyFusedLM(ToyLM):
    """ToyLM with the patched head's ``all_label_log_probs``: every position's label log-prob off the fused
    kernel, next to the kept positions' logits."""

    def forward(self, input_ids=None, logits_to_keep=None, all_label_log_probs=False, shift_labels=None, **kwargs):
        temperature = kwargs.pop("temperature", 1.0)
        output = super().forward(input_ids=input_ids, logits_to_keep=logits_to_keep, **kwargs)
        if all_label_log_probs:
            labels = shift_labels if shift_labels is not None else torch.roll(input_ids, shifts=-1, dims=-1)
            output.log_probs, _ = FusedLinearForPPO().forward(
                self.emb(input_ids), self.head.weight, labels, temperature
            )
        return output


def make_engine(module, patched_head=False, chunked_flag=False, inference_module=None):
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
    flags = {"use_chunked_lm_head": chunked_flag}
    eng.model_config = SimpleNamespace(
        use_fused_kernels=patched_head, get=lambda key, default=None: flags.get(key, default)
    )
    eng.ulysses_device_mesh = None
    eng.ulysses_parallel_group = None
    eng.ulysses_sequence_parallel_size = 1
    eng.use_ulysses_sp = False
    eng.use_remove_padding = True
    eng._is_offload_param = False
    eng._is_offload_optimizer = False
    eng._autocast_dtype = torch.float32
    eng.scaler = None
    eng._inference_module = inference_module
    eng.mode = None
    return eng


def teacher_module(chunked_teacher):
    """The teacher's model and whether its engine runs the patched head; the same weights either way."""
    return (ToyChunkedLM(seed=1), True) if chunked_teacher else (ToyLM(seed=1), False)


def make_batch(
    include_hinted=True,
    include_unhinted=True,
    loss_agg_mode="token-mean",
    hinted_span=(2, 7),
    contrast=None,
    anchor=False,
    outcome=False,
):
    """Build the turn-mode mini-batch exactly as _maybe_build_self_distillation_batch ships it; ``contrast``
    ("same" or "other" tokens than the hint) adds the hinted turn's contrast sub-row, ``anchor`` its no-note
    sub-row, and ``outcome`` makes the un-hinted row a successful one the outcome term reads whole."""
    rows = []
    if include_hinted:
        # hinted sample: prompt 4, response 12, one hinted turn covering response [2, 7); its
        # teacher row is the prompt, the history, the hint and the scored span
        start, end = hinted_span
        prompt = torch.arange(4, dtype=torch.long) + 1
        resp = torch.arange(12, dtype=torch.long) + 5
        hint_ids = torch.tensor([50, 51, 52], dtype=torch.long)
        body = torch.cat([resp[:start], hint_ids, resp[start:end]])
        seq = torch.cat([prompt, body])
        sub_row = SubRow(seq.shape[0], body.shape[0], start + hint_ids.shape[0], start, end)
        contrast_rows = []
        if contrast is not None:
            contrast_ids = hint_ids if contrast == "same" else torch.tensor([60, 61], dtype=torch.long)
            contrast_body = torch.cat([resp[:start], contrast_ids, resp[start:end]])
            contrast_seq = torch.cat([prompt, contrast_body])
            contrast_rows = [
                SubRow(contrast_seq.shape[0], contrast_body.shape[0], start + contrast_ids.shape[0], start, end)
            ]
            seq = torch.cat([seq, contrast_seq])
        anchor_rows = []
        if anchor:
            anchor_seq = torch.cat([prompt, resp[:end]])
            anchor_rows = [SubRow(anchor_seq.shape[0], end, start, start, end)]
            seq = torch.cat([seq, anchor_seq])
        meta = pack([sub_row], contrast_rows, anchor_rows)
        sd_mask = torch.zeros(12)
        sd_mask[start:end] = 1.0
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
        "row_id": njt([torch.tensor([i], dtype=torch.long) for i in range(len(rows))]),
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
    if loss_agg_mode == "traj-mean-token-mean":
        # one row per trajectory here, so the raw shares are 1 for a hinted row and 0 otherwise,
        # and the trainer's denominator is the number of hinted trajectories, clamped to 1
        shares = [float(r["sd_mask"].sum() > 0) for r in rows]
        tensor_dict["trace_weight"] = njt([torch.tensor([w]) for w in shares])
        non_tensor_dict["global_batch_size"] = max(int(sum(shares)), 1)
    if outcome:
        success = [float(r["sd_mask"].sum() == 0) for r in rows]
        tensor_dict["outcome_weight"] = njt([torch.tensor([w]) for w in success])
        tensor_dict["outcome_mask"] = njt([m.float() * w for m, w in zip(response_mask, success, strict=True)])
        non_tensor_dict["outcome_trajectories"] = int(sum(success))
    return tu.get_tensordict(tensor_dict=tensor_dict, non_tensor_dict=non_tensor_dict)


def run_update(
    data,
    loss_agg_mode="token-mean",
    dp_size=None,
    alpha=0.5,
    chunked_teacher=False,
    success_nll_coef=0.0,
    kl_loss_coef=None,
    fused_student=False,
):
    """Mirror ActorRolloutRefWorker.update_actor minus Ray/FSDP/TQ; ``dp_size`` fakes the rank count the
    loss is told, as if every rank held this same batch. ``kl_loss_coef`` turns use_kl_loss on (low_var_kl);
    ``fused_student`` runs the student on the patched head."""
    student_engine = make_engine(ToyFusedLM(seed=0) if fused_student else ToyLM(seed=0), patched_head=fused_student)
    student_engine.engine_config.forward_only = False
    if dp_size is not None:
        student_engine.get_data_parallel_size = lambda: dp_size
    teacher_engine = make_engine(*teacher_module(chunked_teacher))

    worker = SimpleNamespace(
        ref=SimpleNamespace(
            engine=teacher_engine,
            model_config={"use_remove_padding": True},
            engine_config=SimpleNamespace(use_fused_kernels=False),
        )
    )
    teacher_fn = partial(ActorRolloutRefWorker._compute_sdpo_teacher_logps_for_loss, worker)

    actor_config = SimpleNamespace(
        global_batch_info={},
        loss_scale_factor=None,
        loss_agg_mode=loss_agg_mode,
        entropy_coeff=0.0,
        use_kl_loss=kl_loss_coef is not None,
        kl_loss_coef=kl_loss_coef or 0.0,
        kl_loss_type="low_var_kl",
    )
    if kl_loss_coef is not None:
        tu.assign_non_tensor(data, all_label_log_probs=True)
    sdpo_config = SimpleNamespace(
        full_logit_distillation=True,
        distillation_topk=TOPK,
        distillation_add_tail=True,
        alpha=alpha,
        is_clip=2.0,
        success_nll_coef=success_nll_coef,
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


def test_only_a_forward_only_pass_returns_model_output(single_process_group, cpu_ops):
    """The update releases each micro-batch's outputs after its backward; the loss, the metrics
    and the span records still come back."""
    data = make_batch(include_hinted=True, include_unhinted=True)
    student_engine, outputs = run_update(data)
    assert outputs["model_output"] == {}
    assert len(outputs["loss"]) == len(data)
    assert "actor/pg_loss" in outputs["metrics"] and SDPO_SPAN_ROWS_KEY in outputs["metrics"]

    tu.assign_non_tensor(data, distillation_use_topk=False)
    inferred = student_engine.forward_backward_batch(data, None, forward_only=True)
    assert inferred["model_output"]["log_probs"].shape[0] == len(data)


def _pg_loss(outputs):
    metric = outputs["metrics"]["actor/pg_loss"]
    return sum(float(m.aggregate()) for m in (metric if isinstance(metric, list) else [metric]))


def test_traj_mean_all_unhinted_micro_batch_is_a_finite_noop(single_process_group, cpu_ops):
    data = make_batch(include_hinted=False, include_unhinted=True, loss_agg_mode="traj-mean-token-mean")
    assert data["global_batch_size"] == 1
    student_engine, outputs = run_update(data, loss_agg_mode="traj-mean-token-mean")

    assert _pg_loss(outputs) == 0.0
    grads = [p.grad for p in student_engine.module.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) == 0


def test_traj_mean_with_one_hinted_trajectory_matches_token_mean(single_process_group, cpu_ops):
    """One hinted trajectory next to an un-hinted one: the mean over supervised trajectories of
    the per-trajectory token-mean is that trajectory's token-mean, which is what token-mean
    over the mini-batch's supervised tokens computes too. Same loss, same gradient."""
    torch.manual_seed(0)
    data = make_batch(loss_agg_mode="traj-mean-token-mean")
    torch.manual_seed(0)
    reference = make_batch(loss_agg_mode="token-mean")

    traj_engine, traj_out = run_update(data, loss_agg_mode="traj-mean-token-mean")
    token_engine, token_out = run_update(reference, loss_agg_mode="token-mean")

    assert _pg_loss(traj_out) == pytest.approx(_pg_loss(token_out))
    for g_traj, g_tok in zip(traj_engine.module.parameters(), token_engine.module.parameters(), strict=True):
        assert torch.allclose(g_traj.grad, g_tok.grad)
    assert sum(p.grad.abs().sum() for p in traj_engine.module.parameters()) > 0


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


def _score(data, topk_indices, chunked_teacher=False):
    teacher_engine = make_engine(*teacher_module(chunked_teacher))
    worker = SimpleNamespace(
        ref=SimpleNamespace(
            engine=teacher_engine,
            model_config={"use_remove_padding": True},
            engine_config=SimpleNamespace(use_fused_kernels=False),
        )
    )
    tu.assign_non_tensor(data, dp_size=1)
    result = ActorRolloutRefWorker._compute_sdpo_teacher_logps_for_loss(
        worker, data=data, student_topk_indices=topk_indices, return_all_logps=False
    )
    _score.module = teacher_engine.module
    return result


def test_the_contrast_subrow_is_scored_at_the_students_indices_apart_from_the_hint_one(single_process_group, cpu_ops):
    topk = torch.randint(0, VOCAB, (2, 12, TOPK), generator=torch.Generator().manual_seed(0))
    plain = _score(make_batch(), topk)
    other = _score(make_batch(contrast="other"), topk)
    same = _score(make_batch(contrast="same"), topk)

    assert "contrast_topk_log_probs" not in plain
    for result in (other, same):
        for key in ("teacher_log_probs", "teacher_topk_log_probs"):
            torch.testing.assert_close(result[key], plain[key], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        same["contrast_topk_log_probs"][0], same["teacher_topk_log_probs"][0], atol=1e-6, rtol=1e-6
    )

    contrast = other["contrast_topk_log_probs"]
    assert contrast.shape == (2, 12, TOPK)
    assert contrast[0, 2:7].abs().sum() > 0, "the contrast sub-row's span is scored"
    assert contrast[0, :2].abs().sum() == 0 and contrast[0, 7:].abs().sum() == 0 and contrast[1].abs().sum() == 0
    assert not torch.allclose(contrast[0, 2:7], other["teacher_topk_log_probs"][0, 2:7])


@pytest.mark.parametrize("anchor", [False, True])
def test_a_rank_without_contrast_rows_still_returns_the_contrast_output(
    single_process_group, cpu_ops, monkeypatch, anchor
):
    """Another rank holding contrast (and anchor) sub-rows makes this rank's loss report the same metric
    keys: its contrast and anchor outputs are zeros, at no hinted position."""
    real = dist.all_reduce

    def other_rank_has_contrast(tensor, op=None, group=None):
        real(tensor, op=op, group=group)
        if tensor.shape == (3,):  # the (calls, has contrast, has anchor) triple
            tensor[1] = 1
            tensor[2] = int(anchor)

    monkeypatch.setattr(dist, "all_reduce", other_rank_has_contrast)
    result = _score(make_batch(include_hinted=False, include_unhinted=True), torch.zeros(1, 9, TOPK, dtype=torch.long))
    keys = ("contrast_topk_log_probs", "anchor_topk_log_probs") if anchor else ("contrast_topk_log_probs",)
    for key in keys:
        assert result[key].shape == (1, 9, TOPK)
        assert result[key].abs().sum() == 0
    assert anchor or "anchor_topk_log_probs" not in result
    assert result["teacher_topk_log_probs"][0, 0].abs().sum() > 0


def test_the_anchor_subrow_is_the_teacher_on_the_turn_with_no_note(single_process_group, cpu_ops):
    """The anchor output is the teacher scoring the hinted span with nothing spliced, at the student's indices,
    and leaves the hint and contrast outputs as they were."""
    topk = torch.randint(0, VOCAB, (2, 12, TOPK), generator=torch.Generator().manual_seed(0))
    contrast = _score(make_batch(contrast="other"), topk)
    anchored = _score(make_batch(contrast="other", anchor=True), topk)
    for key in ("teacher_log_probs", "teacher_topk_log_probs", "contrast_topk_log_probs"):
        torch.testing.assert_close(anchored[key], contrast[key], atol=1e-6, rtol=1e-6)

    # prompt (4) + response[:7]: the span's tokens sit at 6..10, scored from the logits at 5..9
    no_note = make_batch(include_unhinted=False)["input_ids"].values()[:11]
    with torch.no_grad():
        logits = _score.module(no_note.unsqueeze(0)).logits[0]
    expected = torch.log_softmax(logits.float(), -1)[5:10].gather(-1, topk[0, 2:7])
    torch.testing.assert_close(anchored["anchor_topk_log_probs"][0, 2:7], expected, atol=1e-5, rtol=1e-5)
    assert anchored["anchor_topk_log_probs"][0, :2].abs().sum() == 0
    assert anchored["anchor_topk_log_probs"][0, 7:].abs().sum() == 0
    assert anchored["anchor_topk_log_probs"][1].abs().sum() == 0


@pytest.mark.parametrize("remove_padding", [True, False], ids=["rmpad", "padded"])
@pytest.mark.parametrize("contrast", [None, "other"])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_the_chunked_teacher_head_scores_what_the_logits_path_scores(
    single_process_group, cpu_ops, remove_padding, contrast, temperature
):
    """Same teacher weights, same spliced sub-rows, same student ids: the head that never builds the span's
    logits returns the label and top-k log-probs the extractor reads off them, on both layouts."""
    topk = torch.randint(0, VOCAB, (2, 12, TOPK), generator=torch.Generator().manual_seed(0))

    def score(chunked):
        data = make_batch(contrast=contrast)
        tu.assign_non_tensor(data, use_remove_padding=remove_padding, temperature=temperature)
        return _score(data, topk, chunked_teacher=chunked), _score.module

    eager, eager_module = score(False)
    chunked, chunked_module = score(True)

    assert not isinstance(eager_module, ToyChunkedLM) and chunked_module.scored
    # one call per sub-row (the hinted turn, its contrast, the stub), each scoring its span's ids
    assert [shape for shape, _, _ in chunked_module.scored] == [(5, TOPK)] * (2 if contrast else 1) + [(1, TOPK)]
    assert {chunk for _, chunk, _ in chunked_module.scored} == {512}
    assert all(t == pytest.approx(temperature) for _, _, t in chunked_module.scored)
    for key in ("teacher_log_probs", "teacher_topk_log_probs") + (("contrast_topk_log_probs",) if contrast else ()):
        torch.testing.assert_close(chunked[key], eager[key], atol=1e-5, rtol=1e-5)
    assert chunked["teacher_topk_log_probs"][0, 2:7].abs().sum() > 0


@pytest.mark.parametrize(
    ("engine_flags", "scores_by_chunks"),
    [
        (dict(patched_head=False), False),
        (dict(patched_head=True), True),
        (dict(chunked_flag=True), True),
        (dict(patched_head=True, inference_module=True), False),
    ],
    ids=["stock forward", "fused head", "chunked head", "inference module"],
)
def test_the_chunked_teacher_head_is_only_used_where_the_head_is_patched(
    single_process_group, cpu_ops, engine_flags, scores_by_chunks
):
    """Without the patched forward, or behind an inference module that mixes logits, the model would be
    handed a keyword it does not know: the teacher reads the logits instead."""
    topk = torch.randint(0, VOCAB, (2, 12, TOPK), generator=torch.Generator().manual_seed(0))
    flags = {**engine_flags, "inference_module": ToyLM(seed=1) if engine_flags.get("inference_module") else None}
    teacher_engine = make_engine(ToyChunkedLM(seed=1), **flags)
    worker = SimpleNamespace(
        ref=SimpleNamespace(
            engine=teacher_engine,
            model_config={"use_remove_padding": True},
            engine_config=SimpleNamespace(use_fused_kernels=False),
        )
    )
    data = make_batch()
    tu.assign_non_tensor(data, dp_size=1)
    result = ActorRolloutRefWorker._compute_sdpo_teacher_logps_for_loss(
        worker, data=data, student_topk_indices=topk, return_all_logps=False
    )
    assert bool(teacher_engine.module.scored) == scores_by_chunks
    assert result["teacher_topk_log_probs"][0, 2:7].abs().sum() > 0


@pytest.mark.parametrize("remove_padding", [True, False], ids=["rmpad", "padded"])
def test_the_update_is_the_same_with_the_chunked_teacher_head(single_process_group, cpu_ops, remove_padding):
    """Through the student's loss and backward: the contrast teacher's target, its loss and the student's
    gradient do not change with the head the teacher scores through."""

    def update(chunked):
        torch.manual_seed(0)
        data = make_batch(contrast="other")
        tu.assign_non_tensor(data, use_remove_padding=remove_padding)
        return run_update(data, alpha=1.0, chunked_teacher=chunked)

    eager_engine, eager = update(False)
    chunked_engine, chunked = update(True)

    assert _pg_loss(chunked) == pytest.approx(_pg_loss(eager), rel=1e-4)
    assert sum(g.abs().sum() for g in _grads(chunked_engine)) > 1e-3
    for g_chunked, g_eager in zip(_grads(chunked_engine), _grads(eager_engine), strict=True):
        torch.testing.assert_close(g_chunked, g_eager, atol=1e-5, rtol=1e-4)


def test_the_update_trains_toward_the_difference_of_the_two_teachers(single_process_group, cpu_ops):
    """A contrast identical to the hint leaves the student where it is (no loss, no gradient) while the
    metrics show how far the hint teacher alone would have moved it; a different one trains."""
    torch.manual_seed(0)
    same_engine, same = run_update(make_batch(contrast="same"), alpha=1.0)
    torch.manual_seed(0)
    other_engine, other = run_update(make_batch(contrast="other"), alpha=1.0)

    assert abs(_pg_loss(same)) < 1e-6
    assert all(torch.isfinite(g).all() for g in _grads(same_engine))
    assert sum(g.abs().sum() for g in _grads(same_engine)) < 1e-5
    assert all(torch.isfinite(g).all() for g in _grads(other_engine))
    assert sum(g.abs().sum() for g in _grads(other_engine)) > 1e-3

    tokens = other["metrics"]["self_distillation/supervised_tokens__sum"].aggregate()
    assert tokens == 5
    for outputs in (same, other):
        assert outputs["metrics"]["self_distillation/contrast_bins__sum"].aggregate() == tokens * (TOPK + 1)
    kl_hint = same["metrics"]["self_distillation/kl_hint__sum"].aggregate()
    assert kl_hint > 0 and other["metrics"]["self_distillation/kl_hint__sum"].aggregate() == pytest.approx(kl_hint)
    assert same["metrics"]["self_distillation/kl_target__sum"].aggregate() == pytest.approx(0.0, abs=1e-5)
    assert other["metrics"]["self_distillation/kl_target__sum"].aggregate() > 1e-4

    summed = finalize_ratio_metrics({k: v.aggregate() for k, v in other["metrics"].items() if k.endswith("__sum")})
    for name in ("kl_hint", "kl_target", "contrast_floor_masked"):
        assert f"self_distillation/{name}" in summed
    assert not any(key.endswith("__sum") for key in summed)


def test_the_anchored_update_trains_toward_the_no_note_teacher_plus_the_shift(single_process_group, cpu_ops):
    """End to end with the three sub-rows: a contrast equal to the hint leaves the no-note teacher as the
    target (the toy student is not it, so it trains); the entropy metrics come out against both."""
    torch.manual_seed(0)
    same_engine, same = run_update(make_batch(contrast="same", anchor=True), alpha=1.0)
    torch.manual_seed(0)
    other_engine, other = run_update(make_batch(contrast="other", anchor=True), alpha=1.0)
    for engine in (same_engine, other_engine):
        grads = _grads(engine)
        assert all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads) > 1e-3
    assert not all(torch.equal(a, b) for a, b in zip(_grads(same_engine), _grads(other_engine), strict=True))

    summed = finalize_ratio_metrics({k: v.aggregate() for k, v in other["metrics"].items() if k.endswith("__sum")})
    for name in ("target_entropy_minus_student", "target_entropy_minus_anchor", "kl_target"):
        assert math.isfinite(summed[f"self_distillation/{name}"])


def _outcome_update(coef, outcome=True):
    torch.manual_seed(0)
    data = make_batch(loss_agg_mode="traj-mean-token-mean", outcome=outcome)
    return run_update(data, loss_agg_mode="traj-mean-token-mean", alpha=1.0, success_nll_coef=coef)


def test_a_zero_outcome_coefficient_leaves_the_update_as_it_was(single_process_group, cpu_ops):
    """The trainer ships no outcome field at coefficient 0; the loss also ignores them if they are there."""
    today_engine, today = _outcome_update(0.0, outcome=False)
    zero_engine, zero = _outcome_update(0.0)
    for a, b in zip(_grads(today_engine), _grads(zero_engine), strict=True):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=0)
    assert _pg_loss(today) == pytest.approx(_pg_loss(zero), abs=1e-7)
    assert not [k for k in zero["metrics"] if k.startswith("outcome/") or "grad_proxy" in k]


def test_the_outcome_term_adds_its_nll_to_the_distillation_loss(single_process_group, cpu_ops):
    """The successful row's NLL enters with its coefficient, linearly; its metrics are its own tokens."""
    base_engine, _ = _outcome_update(0.0)
    one_engine, one = _outcome_update(0.3)
    two_engine, _ = _outcome_update(0.6)
    base, g1, g2 = _grads(base_engine), _grads(one_engine), _grads(two_engine)
    assert sum((a - b).abs().sum() for a, b in zip(g1, base, strict=True)) > 1e-4
    for b, a1, a2 in zip(base, g1, g2, strict=True):
        torch.testing.assert_close(a2 - b, 2 * (a1 - b), atol=1e-6, rtol=1e-4)

    metrics = {k: v.aggregate() for k, v in one["metrics"].items() if hasattr(v, "aggregate")}
    assert metrics["outcome/tokens__sum"] == 9 and metrics["outcome/rows__sum"] == 1
    # the student's own NLL on the successful row: prompt 3 + response 9, the response scored from 2..10
    student = ToyLM(seed=0)
    ids = make_batch()["input_ids"].unbind()[1]
    with torch.no_grad():
        log_probs = torch.log_softmax(student(ids.unsqueeze(0)).logits[0], -1)
    nll = -log_probs[2:11].gather(-1, ids[3:12].unsqueeze(-1)).mean()
    ratios = finalize_ratio_metrics({k: v for k, v in metrics.items() if k.endswith("__sum")})
    assert ratios["outcome/nll_per_token"] == pytest.approx(float(nll), rel=1e-5)
    assert metrics["outcome/nll_loss"] == pytest.approx(float(nll), rel=1e-5), "one trajectory, weight 1"
    for key in ("outcome/grad_proxy_nll", "self_distillation/grad_proxy_sd"):
        assert ratios[key] > 0
    assert metrics["outcome/grad_proxy_nll_term"] > 0 and metrics["self_distillation/grad_proxy_sd_term"] > 0


def with_reference(data, shift=0.0, kl_trajectories=1):
    """The toy student's own log-prob of each response token as the reference, moved by ``shift``; the trainer's
    count of the trajectories the KL term reads (None ships none)."""
    student = ToyLM(seed=0)
    rows = []
    with torch.no_grad():
        for ids, prompt in zip(data["input_ids"].unbind(), data["prompts"].unbind(), strict=True):
            log_probs = torch.log_softmax(student(ids.unsqueeze(0)).logits[0], -1)
            per_position = log_probs[:-1].gather(-1, ids[1:].unsqueeze(-1)).squeeze(-1)
            rows.append(per_position[prompt.shape[0] - 1 :] + shift)
    data["ref_log_prob"] = njt(rows)
    if kl_trajectories is not None:
        tu.assign_non_tensor(data, kl_trajectories=kl_trajectories)
    return data


def _kl_update(kl_loss_coef, shift=0.0, outcome=False):
    torch.manual_seed(0)
    data = with_reference(
        make_batch(loss_agg_mode="traj-mean-token-mean", outcome=outcome), shift, kl_trajectories=2 if outcome else 1
    )
    return run_update(
        data,
        loss_agg_mode="traj-mean-token-mean",
        alpha=1.0,
        success_nll_coef=0.3 if outcome else 0.0,
        kl_loss_coef=kl_loss_coef,
        fused_student=True,
    )


def _summed(outputs):
    return {k: v.aggregate() for k, v in outputs["metrics"].items() if hasattr(v, "aggregate")}


def test_a_kl_off_update_is_todays_bit_for_bit(single_process_group, cpu_ops):
    """The patched student and a reference in the batch change nothing while use_kl_loss is off."""
    torch.manual_seed(0)
    today_engine, today = run_update(
        make_batch(loss_agg_mode="traj-mean-token-mean"), loss_agg_mode="traj-mean-token-mean", alpha=1.0
    )
    off_engine, off = _kl_update(None)
    for a, b in zip(_grads(today_engine), _grads(off_engine), strict=True):
        assert torch.equal(a, b)
    assert _pg_loss(today) == _pg_loss(off)
    assert set(today["metrics"]) == set(off["metrics"])


def test_the_kl_term_is_zero_where_the_student_is_the_reference(single_process_group, cpu_ops):
    off_engine, _ = _kl_update(None)
    on_engine, on = _kl_update(0.5)
    assert _summed(on)["kl_loss"] == pytest.approx(0.0, abs=1e-6)
    for a, b in zip(_grads(off_engine), _grads(on_engine), strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=0)


def test_the_kl_term_pulls_the_sampled_tokens_toward_the_reference(single_process_group, cpu_ops):
    """A reference one nat below the student on every sampled token: a step against the term's gradient lowers the
    hinted row's log-probs, and only its rows' tokens are read."""
    off_engine, _ = _kl_update(None, shift=-1.0)
    on_engine, on = _kl_update(0.5, shift=-1.0)
    kl_grads = [b - a for a, b in zip(_grads(off_engine), _grads(on_engine), strict=True)]

    student = ToyLM(seed=0)
    data = make_batch(loss_agg_mode="traj-mean-token-mean")
    ids, prompt, mask = (data[key].unbind()[0] for key in ("input_ids", "prompts", "response_mask"))
    log_probs = torch.log_softmax(student(ids.unsqueeze(0)).logits[0], -1)[:-1]
    log_probs = log_probs.gather(-1, ids[1:].unsqueeze(-1)).squeeze(-1)[prompt.shape[0] - 1 :]
    (log_probs * mask).sum().backward()
    assert sum((g * p.grad).sum() for g, p in zip(kl_grads, student.parameters(), strict=True)) > 0
    assert _summed(on)["kl_loss"] > 0


def test_the_kl_term_adds_linearly_with_its_coefficient(single_process_group, cpu_ops):
    base_engine, _ = _kl_update(None, shift=-0.5)
    one_engine, _ = _kl_update(0.2, shift=-0.5)
    two_engine, _ = _kl_update(0.4, shift=-0.5)
    base, g1, g2 = _grads(base_engine), _grads(one_engine), _grads(two_engine)
    assert sum((a - b).abs().sum() for a, b in zip(g1, base, strict=True)) > 1e-4
    for b, a1, a2 in zip(base, g1, g2, strict=True):
        torch.testing.assert_close(a2 - b, 2 * (a1 - b), atol=1e-6, rtol=1e-4)


def test_the_kl_term_reads_every_response_token_of_the_rows_it_weighs(single_process_group, cpu_ops):
    _, on = _kl_update(0.3, shift=-0.5)
    metrics = _summed(on)
    # the hinted row's 12 response tokens but the masked first one; the un-hinted row carries no weight
    assert metrics["self_distillation/kl_tokens__sum"] == 11
    assert metrics["self_distillation/supervised_tokens__sum"] == 5
    ratios = finalize_ratio_metrics({k: v for k, v in metrics.items() if k.endswith("__sum")})
    assert ratios["self_distillation/kl_tokens"] == 11 and ratios["self_distillation/grad_proxy_kl"] > 0
    assert metrics["self_distillation/grad_proxy_kl_term"] > 0 and metrics["self_distillation/grad_proxy_sd_term"] > 0
    coef = on["metrics"]["kl_coef"]
    assert set(coef if isinstance(coef, list) else [coef]) == {0.3}, "one per micro-batch, averaged downstream"


def test_the_kl_term_reads_the_outcome_rows_too(single_process_group, cpu_ops):
    _, on = _kl_update(0.3, shift=-0.5, outcome=True)
    metrics = _summed(on)
    assert metrics["self_distillation/kl_tokens__sum"] == 11 + 9
    assert metrics["outcome/tokens__sum"] == 9 and metrics["outcome/grad_proxy_nll_term"] > 0
    assert metrics["kl_loss"] > 0


def test_a_kl_update_without_its_inputs_is_refused(single_process_group, cpu_ops):
    def update(data, fused_student=True):
        return run_update(
            data, loss_agg_mode="traj-mean-token-mean", alpha=1.0, kl_loss_coef=0.1, fused_student=fused_student
        )

    with pytest.raises(ValueError, match="ref_log_prob"):
        update(make_batch(loss_agg_mode="traj-mean-token-mean"))
    with pytest.raises(ValueError, match="kl_trajectories"):
        update(with_reference(make_batch(loss_agg_mode="traj-mean-token-mean"), kl_trajectories=None))
    with pytest.raises(ValueError, match="use_fused_kernels"):
        update(with_reference(make_batch(loss_agg_mode="traj-mean-token-mean")), fused_student=False)


def test_per_span_export_names_its_span_and_matches_the_batch_metrics(single_process_group, cpu_ops):
    """The batch holds exactly one supervised span, so every per-span aggregate has to come out
    equal to the batch metric computed independently over the same positions."""
    data = make_batch(include_hinted=True, include_unhinted=True)
    _, outputs = run_update(data)
    metrics = outputs["metrics"]

    (span,) = metrics[SDPO_SPAN_ROWS_KEY]
    assert (span["row_id"], span["start"], span["end"], span["supervised_tokens"]) == (0, 2, 7, 5)

    tokens = metrics["self_distillation/supervised_tokens__sum"].aggregate()
    assert tokens == span["supervised_tokens"]
    assert span["absgap_mean"] == pytest.approx(metrics["self_distillation/absgap__sum"].aggregate() / tokens, rel=1e-5)
    summed = finalize_ratio_metrics({k: v.aggregate() for k, v in metrics.items() if k.endswith("__sum")})
    # the step median is read off a histogram with bins 1.18x wide, the span's is exact
    assert span["loss_p50"] == pytest.approx(summed["self_distillation/loss_p50"], rel=0.2)
    assert span["teacher_prefers_other_token"] == pytest.approx(
        metrics["self_distillation/teacher_prefers_other__sum"].aggregate() / tokens
    )


def test_summed_counts_are_global_over_dp(single_process_group, cpu_ops):
    """aggregate_dp averages SUM metrics over ranks, so each rank's '__sum' carries the dp factor:
    two ranks holding this batch count its tokens twice, while the ratios stay what they were."""
    one = run_update(make_batch(include_hinted=True, include_unhinted=True))[1]["metrics"]
    two = run_update(make_batch(include_hinted=True, include_unhinted=True), dp_size=2)[1]["metrics"]
    key = "self_distillation/supervised_tokens__sum"
    assert two[key].aggregate() == 2 * one[key].aggregate()
    ratio = "self_distillation/gap__sum"
    assert two[ratio].aggregate() / two[key].aggregate() == pytest.approx(one[ratio].aggregate() / one[key].aggregate())


def test_nothing_is_exported_without_supervision_or_row_ids(single_process_group, cpu_ops):
    _, unhinted = run_update(make_batch(include_hinted=False, include_unhinted=True))
    assert SDPO_SPAN_ROWS_KEY not in unhinted["metrics"]

    data = make_batch(include_hinted=True, include_unhinted=True)
    del data["row_id"]
    _, no_ids = run_update(data)
    assert SDPO_SPAN_ROWS_KEY not in no_ids["metrics"]
    assert no_ids["metrics"]["self_distillation/supervised_tokens__sum"].aggregate() == 5


def test_span_rows_ride_the_dp_metric_channel_as_records(single_process_group, cpu_ops):
    """The channel out of the update is scalar-only: the records travel as a plain list, which
    the dp allgather and the mini-batch flattening (train_mini_batch) carry through untouched."""
    _, outputs = run_update(make_batch(include_hinted=True, include_unhinted=True))
    rows = outputs["metrics"][SDPO_SPAN_ROWS_KEY]
    assert not isinstance(rows, Metric), "a Metric would be averaged over the dp ranks instead"

    gathered = allgather_dict_into_dict({SDPO_SPAN_ROWS_KEY: rows})[SDPO_SPAN_ROWS_KEY]
    assert list(chain.from_iterable(gathered)) == rows
def _grads(engine):
    return [p.grad for p in engine.module.parameters()]


def _lens(data, key):
    return data[key].offsets().diff().tolist()


def test_truncation_after_the_last_supervised_token_changes_nothing(single_process_group, cpu_ops):
    """The differential the cut has to pass: identical loss, identical gradients, identical
    trace weight, on a batch whose hinted row carries five unsupervised tokens of tail."""
    torch.manual_seed(0)
    whole = make_batch(loss_agg_mode="traj-mean-token-mean")
    torch.manual_seed(0)
    cut = make_batch(loss_agg_mode="traj-mean-token-mean")

    kept_fraction = truncate_rows_after_last_supervised_token(cut)

    # hinted row: response 12 -> 7 (span ends at 7), prompt 4 so the sequence goes 16 -> 11;
    # un-hinted row: untouched. 21 response tokens in, 16 out.
    assert _lens(whole, "responses") == [12, 9] and _lens(cut, "responses") == [7, 9]
    assert _lens(whole, "input_ids") == [16, 12] and _lens(cut, "input_ids") == [11, 12]
    assert kept_fraction == pytest.approx(16 / 21)

    whole_engine, whole_out = run_update(whole, loss_agg_mode="traj-mean-token-mean")
    cut_engine, cut_out = run_update(cut, loss_agg_mode="traj-mean-token-mean")

    assert _pg_loss(cut_out) == pytest.approx(_pg_loss(whole_out))
    for g_cut, g_whole in zip(_grads(cut_engine), _grads(whole_engine), strict=True):
        assert torch.allclose(g_cut, g_whole)
    assert sum(g.abs().sum() for g in _grads(cut_engine)) > 0


def test_truncation_leaves_the_teacher_rows_and_trace_weight_alone(single_process_group, cpu_ops):
    """Cutting the tail cannot move an earlier offset, so the spliced teacher geometry and the
    per-row supervised-token count the trajectory mean divides by must come through untouched."""
    torch.manual_seed(0)
    whole = make_batch(loss_agg_mode="traj-mean-token-mean")
    torch.manual_seed(0)
    cut = make_batch(loss_agg_mode="traj-mean-token-mean")
    truncate_rows_after_last_supervised_token(cut)

    for key in ("teacher_input_ids", "teacher_seq_meta", "trace_weight", "prompts"):
        assert _lens(cut, key) == _lens(whole, key)
        for row_cut, row_whole in zip(cut[key].unbind(), whole[key].unbind(), strict=True):
            assert torch.equal(row_cut, row_whole)
    # every supervised position survives the cut
    cut_masks, whole_masks = cut["self_distillation_mask"].unbind(), whole["self_distillation_mask"].unbind()
    for row_cut, row_whole in zip(cut_masks, whole_masks, strict=True):
        assert row_cut.sum() == row_whole.sum()


def test_a_row_ending_on_its_last_supervised_token_is_untouched(single_process_group, cpu_ops):
    data = make_batch(include_unhinted=False, hinted_span=(2, 12))
    before = _lens(data, "responses")
    assert truncate_rows_after_last_supervised_token(data) == 1.0
    assert _lens(data, "responses") == before


def test_a_row_with_no_supervised_token_is_untouched(single_process_group, cpu_ops):
    data = make_batch(include_hinted=False)
    before_resp, before_seq = _lens(data, "responses"), _lens(data, "input_ids")
    assert truncate_rows_after_last_supervised_token(data) == 1.0
    assert _lens(data, "responses") == before_resp and _lens(data, "input_ids") == before_seq

    torch.manual_seed(0)
    reference = make_batch(include_hinted=False)
    cut_engine, _ = run_update(data)
    ref_engine, _ = run_update(reference)
    for g_cut, g_ref in zip(_grads(cut_engine), _grads(ref_engine), strict=True):
        assert torch.equal(g_cut, g_ref)


def test_truncation_leaves_a_successful_row_whole_for_the_outcome_term(single_process_group, cpu_ops):
    """The outcome rows carry no supervised token, so the cut leaves them, their outcome mask and weight
    whole; the hinted row's (empty) outcome mask is cut with its response; the update is unchanged."""
    torch.manual_seed(0)
    whole = make_batch(loss_agg_mode="traj-mean-token-mean", outcome=True)
    torch.manual_seed(0)
    cut = make_batch(loss_agg_mode="traj-mean-token-mean", outcome=True)
    truncate_rows_after_last_supervised_token(cut)

    assert _lens(cut, "responses") == [7, 9] and _lens(cut, "outcome_mask") == [7, 9]
    assert torch.equal(cut["outcome_mask"].unbind()[1], whole["outcome_mask"].unbind()[1])
    assert cut["outcome_mask"].unbind()[1].sum() == 9
    for row_cut, row_whole in zip(cut["outcome_weight"].unbind(), whole["outcome_weight"].unbind(), strict=True):
        assert torch.equal(row_cut, row_whole)

    whole_engine, whole_out = run_update(whole, loss_agg_mode="traj-mean-token-mean", success_nll_coef=0.3)
    cut_engine, cut_out = run_update(cut, loss_agg_mode="traj-mean-token-mean", success_nll_coef=0.3)
    assert _pg_loss(cut_out) == pytest.approx(_pg_loss(whole_out))
    for g_cut, g_whole in zip(_grads(cut_engine), _grads(whole_engine), strict=True):
        assert torch.allclose(g_cut, g_whole)


def test_a_field_off_the_response_grid_on_one_row_is_reported_not_skipped(single_process_group, cpu_ops):
    """One row of old_log_probs longer than its response: skipping the field would cut the student
    rows and leave old_log_probs whole, which is the misalignment the IS ratio then trips over."""
    torch.manual_seed(0)
    data = make_batch(loss_agg_mode="traj-mean-token-mean")
    rows = list(data["old_log_probs"].unbind())
    rows[1] = torch.cat([rows[1], torch.zeros(2)])
    data["old_log_probs"] = torch.nested.nested_tensor(rows, layout=torch.jagged)

    with pytest.raises(RuntimeError, match=r"'old_log_probs' is on the response grid for 1 of 2 rows") as err:
        truncate_rows_after_last_supervised_token(data)
    assert "row 1: old_log_probs 11, responses 9" in str(err.value)


def test_a_constant_length_field_is_on_no_grid_even_where_a_pad_row_matches_it(single_process_group, cpu_ops):
    """A per-row placeholder of length 1 coincides with a 1-token pad row's response, and is still
    no response-grid field: the cut leaves it alone instead of reporting it."""
    torch.manual_seed(0)
    data = make_batch(loss_agg_mode="traj-mean-token-mean")
    rows = list(data["responses"].unbind())
    rows[1] = rows[1][:1]
    data["responses"] = torch.nested.nested_tensor(rows, layout=torch.jagged)
    data["dummy_tensor"] = torch.nested.nested_tensor([torch.zeros(1), torch.zeros(1)], layout=torch.jagged)

    _raise_if_partly_on_the_grid(data, "dummy_tensor", [1, 1], _lens(data, "responses"), _lens(data, "input_ids"))


def test_a_causal_forward_does_not_see_the_tail_it_is_cut_from():
    """Why the cut is exact rather than an approximation, on the attention itself: under a causal
    mask, dropping a suffix leaves every earlier position's output bit-identical, so the student's
    log-probs on the supervised spans are what the whole row would have produced."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 16, 8, dtype=torch.float64) for _ in range(3))
    keep = 9

    whole = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    cut = torch.nn.functional.scaled_dot_product_attention(
        q[:, :, :keep], k[:, :, :keep], v[:, :, :keep], is_causal=True
    )
    assert torch.equal(cut, whole[:, :, :keep])
