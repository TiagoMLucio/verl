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
"""Span-only teacher lm_head: computing values only at ``turn_keep_positions`` and
scattering them back to the packed layout must reproduce the full-computation grid
(including the -1 next-token shift shared with ``no_padding_2_padding``). Span-less
rows (un-hinted / padding stubs) are excluded from the teacher and keep one dummy
student position under the hints-only contract."""

import torch
from tensordict import TensorDict

from verl.workers.utils.padding import no_padding_2_padding
from verl.workers.utils.sdpo import (
    explode_turn_teacher_rows,
    response_keep_positions,
    scatter_turn_teacher_outputs,
    turn_keep_positions,
)


def njt(rows):
    return torch.nested.nested_tensor(rows, layout=torch.jagged)


def make_rows():
    # hinted sample: body 200, spans over response grid [10, 40) and [120, 200)
    hinted = {
        "responses": torch.arange(200, dtype=torch.long),
        "response_mask": torch.ones(200),
        "teacher_input_ids": torch.arange(260, dtype=torch.long),
        "teacher_seq_meta": torch.tensor([200, 10, 10, 40, 120, 120, 200], dtype=torch.int64),
    }
    # degenerate stub (un-hinted or padding row): 1-token teacher body, zero mask
    stub = {
        "responses": torch.tensor([11], dtype=torch.long),
        "response_mask": torch.zeros(1),
        "teacher_input_ids": torch.tensor([8, 11], dtype=torch.long),
        "teacher_seq_meta": torch.tensor([1, 0, 0, 1], dtype=torch.int64),
    }
    return hinted, stub


def test_keep_positions_match_consumed_positions():
    hinted, stub = make_rows()
    sub_seqs, sub_resps, _, parents, spans = explode_turn_teacher_rows(
        teacher_input_ids=njt([hinted["teacher_input_ids"], stub["teacher_input_ids"]]),
        teacher_seq_meta=njt([hinted["teacher_seq_meta"], stub["teacher_seq_meta"]]),
        responses=njt([hinted["responses"], stub["responses"]]),
        response_mask=njt([hinted["response_mask"], stub["response_mask"]]),
    )
    # every row keeps a teacher sub-row (dp-collective lockstep); the stub's is 1 token
    assert parents == [0, 1]
    keep = turn_keep_positions(sub_seqs, sub_resps, spans)

    seq_lens = sub_seqs.offsets().diff()
    body_lens = sub_resps.offsets().diff()

    # full-path values: position index over the packed layout
    offsets = sub_seqs.offsets()
    total_nnz = int(offsets[-1])
    values_full = torch.arange(total_nnz, dtype=torch.float32)

    teacher_td = TensorDict(
        {
            "prompts": njt([seq[: seq_lens[j] - body_lens[j]] for j, seq in enumerate(sub_seqs.unbind())]),
            "responses": sub_resps,
        },
        batch_size=len(parents),
    )
    batch_size, response_length = 2, 200

    def to_grid(packed_values):
        padded = no_padding_2_padding(torch.nested.nested_tensor_from_jagged(packed_values, offsets), teacher_td)
        return scatter_turn_teacher_outputs(padded, parents, spans, batch_size, response_length)

    grid_full = to_grid(values_full)

    # reduced path: compute only at keep positions, scatter back into zeros
    keep_idx = keep.values() + offsets[:-1].repeat_interleave(keep.offsets().diff())
    values_reduced = values_full.new_zeros(total_nnz).index_copy_(0, keep_idx, values_full[keep_idx])
    grid_reduced = to_grid(values_reduced)

    assert torch.equal(grid_full, grid_reduced), "span-only computation must reproduce the consumed grid"
    # sanity: the grid actually carries the span values (scatter is not a no-op)
    assert grid_full[0, 10:40].abs().sum() > 0
    # the stub row only carries its degenerate first-position value (masked in the loss)
    assert grid_full[1, 1:].abs().sum() == 0


def test_meta_without_triples_keeps_dummy_positions():
    """Defensive: a span-less meta still yields a dummy keep position (graph connectivity)
    and a teacher sub-row that scatters nothing."""
    _, stub = make_rows()
    sub_seqs, sub_resps, _, parents, spans = explode_turn_teacher_rows(
        teacher_input_ids=njt([stub["teacher_input_ids"]]),
        teacher_seq_meta=njt([torch.tensor([1], dtype=torch.int64)]),
        responses=njt([stub["responses"]]),
        response_mask=njt([stub["response_mask"]]),
    )
    assert parents == [0] and spans == [[]]
    keep = turn_keep_positions(sub_seqs, sub_resps, spans)
    assert keep.unbind()[0].tolist() == [0]
    grid = scatter_turn_teacher_outputs(torch.ones(1, 1), parents, spans, 1, 1)
    assert grid.abs().sum() == 0


def test_student_keep_positions_cover_masked_grid():
    hinted, stub = make_rows()
    input_ids = njt([torch.arange(250, dtype=torch.long), torch.arange(2, dtype=torch.long)])  # prompt 50 / 1
    responses = njt([hinted["responses"], stub["responses"]])
    meta = njt([hinted["teacher_seq_meta"], stub["teacher_seq_meta"]])

    keep = response_keep_positions(input_ids, responses, meta)

    # span-less row keeps exactly one dummy position (graph connectivity), at its prefix boundary
    assert keep.offsets().diff().tolist() == [110, 1]
    assert keep.unbind()[1].tolist() == [0]

    offsets = input_ids.offsets()
    total_nnz = int(offsets[-1])
    values_full = torch.arange(total_nnz, dtype=torch.float32)
    td = TensorDict(
        {"prompts": njt([torch.arange(50), torch.arange(1)]), "responses": responses},
        batch_size=2,
    )

    def to_grid(packed_values):
        return no_padding_2_padding(torch.nested.nested_tensor_from_jagged(packed_values, offsets), td)

    keep_idx = keep.values() + offsets[:-1].repeat_interleave(keep.offsets().diff())
    values_reduced = values_full.new_zeros(total_nnz).index_copy_(0, keep_idx, values_full[keep_idx])

    span_mask = torch.zeros(2, 200, dtype=torch.bool)
    span_mask[0, 10:40] = span_mask[0, 120:200] = True

    grid_full, grid_reduced = to_grid(values_full), to_grid(values_reduced)
    assert torch.equal(grid_full[span_mask], grid_reduced[span_mask]), (
        "student span-only values must match the full computation on every masked position"
    )
    assert grid_full[span_mask].abs().sum() > 0


def test_single_keep_position_processor_contract():
    """n_keep == 1 (all-unhinted micro): processor outputs keep the leading batch dim, so the
    engine's squeeze(0) + scatter still see (n_keep, k) rows."""
    from types import SimpleNamespace

    from verl.workers.utils.losses import sdpo_ppo_loss

    logits = torch.randn(1, 1, 32)  # (1, n_keep=1, vocab)
    cfg = SimpleNamespace(distillation_topk=5)
    outputs = sdpo_ppo_loss(
        config=None, sdpo_config=cfg, student_logits=logits, data=None, logits_keep_idx=torch.tensor([4])
    )
    assert outputs["topk_logps"].shape == (1, 1, 5)

    v = outputs["topk_logps"].squeeze(0)  # engine unbatching
    total_nnz, keep_idx = 13, torch.tensor([4])
    full = v.new_zeros((total_nnz, *v.shape[1:])).index_copy_(0, keep_idx, v)
    assert full.shape == (13, 5) and full[4].abs().sum() > 0
