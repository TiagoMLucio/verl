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
"""Spliced SDPO teacher rows scored through a cache: every branch, continued from its sample's base
through a transformers cache (attention keys and values, linear-attention recurrent states), scores
its span as a fresh forward of its whole sub-row does, on a hybrid attention and gated delta net model.

The model test needs CUDA: the gated delta net's kernels do."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import verl.workers.utils.sdpo as sdpo_utils
from verl.trainer.ppo.sdpo.teacher_meta import DEGENERATE_META, SubRow, pack, sub_row_ids, unpack_row
from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.utils.sdpo import TeacherCache, continuation_pieces

VOCAB, TOPK = 96, 6


def njt(rows):
    return torch.nested.nested_tensor(rows, layout=torch.jagged)


def test_continuation_pieces_cover_the_range_in_bounded_pieces(monkeypatch):
    monkeypatch.setattr(sdpo_utils, "CONTINUATION_PIECE", 4)
    for start, stop in ((10, 11), (10, 13), (10, 18), (10, 41)):
        pieces = continuation_pieces(start, stop)
        assert pieces[0][0] == start and pieces[-1][1] == stop
        assert all(hi == lo for (_, hi), (lo, _) in zip(pieces, pieces[1:], strict=False))
        assert all(hi - lo < 2 * 4 for lo, hi in pieces)
        assert len(pieces) == 1 or all(hi - lo >= 4 for lo, hi in pieces)


@pytest.fixture(scope="module")
def single_process_group(tmp_path_factory):
    if not dist.is_initialized():
        store_file = tmp_path_factory.mktemp("pg") / "store"
        dist.init_process_group(backend="gloo", init_method=f"file://{store_file}", world_size=1, rank=0)
    yield
    dist.destroy_process_group()


def tiny_qwen35():
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        layer_types=["linear_attention", "full_attention"] * 2,
        num_attention_heads=2,
        num_key_value_heads=1,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
    )
    torch.manual_seed(0)
    return Qwen3_5ForCausalLM(config).to("cuda", torch.float32).eval()


def make_engine(module):
    """An FSDPEngineWithLMHead around an unwrapped module, as the CPU update-path test builds it."""
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


def teacher_batch(remove_padding):
    """A sample with three hinted turns (each with its hint, contrast and no-note sub-rows), the notes
    before a 2-token header, and an un-hinted sample's stub."""
    g = torch.Generator().manual_seed(1)

    def ids(n):
        return torch.randint(0, VOCAB, (n,), generator=g)

    prompt, response = ids(9), ids(40)
    spans = [(4, 10), (16, 21), (30, 37)]
    splits = [prompt.shape[0] + start - 2 for start, _ in spans]
    hints, contrasts = [ids(3), ids(5), ids(2)], [ids(4), ids(1), ids(3)]
    groups = [
        [
            SubRow(split, note.shape[0], start, end)
            for split, note, (start, end) in zip(splits, notes, spans, strict=True)
        ]
        for notes in (hints, contrasts, [ids(0)] * 3)
    ]
    hinted = torch.cat([prompt, response[:37], *hints, *contrasts])
    stub_response = ids(12)
    data = tu.get_tensordict(
        tensor_dict={
            "teacher_input_ids": njt([hinted, torch.cat([prompt[-1:], stub_response[:1]])]),
            "teacher_seq_meta": njt([torch.tensor(pack(9, *groups)), torch.tensor(DEGENERATE_META)]),
            "responses": njt([response, stub_response]),
            "response_mask": njt([torch.ones(40, dtype=torch.int64), torch.ones(12, dtype=torch.int64)]),
            "self_distillation_mask": njt([torch.ones(40), torch.zeros(12)]),
        },
        non_tensor_dict={"temperature": 1.0, "pad_token_id": 0, "use_remove_padding": remove_padding},
    )
    return data, hinted, pack(9, *groups)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the gated delta net's kernels need CUDA")
@pytest.mark.parametrize("remove_padding", [True, False], ids=["rmpad", "padded"])
def test_branches_score_their_spans_as_fresh_subrow_forwards(single_process_group, monkeypatch, remove_padding):
    # the base between two hinted turns is read on in several pieces
    monkeypatch.setattr(sdpo_utils, "CONTINUATION_PIECE", 3)
    model = tiny_qwen35()
    data, hinted, meta = teacher_batch(remove_padding)
    topk = torch.randint(0, VOCAB, (2, 40, TOPK), generator=torch.Generator().manual_seed(2))
    worker = SimpleNamespace(ref=SimpleNamespace(engine=make_engine(model), model_config={}))
    result = ActorRolloutRefWorker._compute_sdpo_teacher_logps_for_loss(worker, data=data, student_topk_indices=topk)

    base, offset, noted = unpack_row(hinted, meta)
    covered = torch.zeros(40, dtype=torch.bool)
    for sub in noted:
        full = sub_row_ids(base, offset, sub).cuda()
        start, end = sub.row.start, sub.row.end
        positions = torch.arange(full.shape[0] - (end - start) - 1, full.shape[0] - 1, device="cuda")
        with torch.no_grad():
            logps = torch.log_softmax(model(full.unsqueeze(0)).logits[0, positions].float(), dim=-1)
        key = (
            "anchor_topk_log_probs"
            if sub.anchor
            else "contrast_topk_log_probs"
            if sub.contrast
            else "teacher_topk_log_probs"
        )
        expected = logps.gather(-1, topk[0, start:end].cuda())
        torch.testing.assert_close(result[key][0, start:end], expected, atol=1e-4, rtol=1e-4)
        if not (sub.contrast or sub.anchor):
            label = logps.gather(-1, full[positions + 1, None]).squeeze(-1)
            torch.testing.assert_close(result["teacher_log_probs"][0, start:end], label, atol=1e-4, rtol=1e-4)
        covered[start:end] = True
    for key in ("teacher_topk_log_probs", "contrast_topk_log_probs", "anchor_topk_log_probs"):
        assert result[key][0, ~covered.cuda()].abs().sum() == 0
        assert result[key][1, 1:].abs().sum() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the gated delta net's kernels need CUDA")
def test_the_teacher_cache_keeps_the_recurrent_states_in_float32():
    """A bf16 model's recurrent states leave the kernel in fp32; the teacher cache keeps them so, where
    transformers' own would round them to the model's dtype at every handover between forwards."""
    model = tiny_qwen35().to(torch.bfloat16)
    cache = TeacherCache(config=model.config)
    with torch.no_grad():
        model(torch.randint(0, VOCAB, (1, 20), device="cuda"), past_key_values=cache, use_cache=True)
    states = [layer.recurrent_states for layer in cache.layers if getattr(layer, "has_previous_state", False)]
    assert states and all(state.dtype == torch.float32 for state in states)
