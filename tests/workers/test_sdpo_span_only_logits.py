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
"""Span-only teacher lm_head: each branch keeps exactly the positions that predict its scored
span (the -1 next-token shift shared with ``no_padding_2_padding``), and their values scatter
back onto the response grid. Un-hinted / padding rows ship the degenerate sub-row so the teacher
still scores them, and keep one dummy student keep position under the hints-only contract."""

import torch
from tensordict import TensorDict

from verl.trainer.ppo.sdpo.teacher_meta import DEGENERATE_META, SubRow, pack
from verl.workers.utils.padding import no_padding_2_padding
from verl.workers.utils.sdpo import response_keep_positions, scatter_turn_teacher_outputs, teacher_branches


def njt(rows):
    return torch.nested.nested_tensor(rows, layout=torch.jagged)


def make_rows():
    # hinted sample: a 60-token prompt, then the response up to the last span; two sub-rows with their
    # notes before the turns whose spans are [10, 40) and [120, 200) on the response grid
    response = torch.arange(200, dtype=torch.long)
    sub_rows = [SubRow(split=68, note_len=3, start=10, end=40), SubRow(split=178, note_len=5, start=120, end=200)]
    hinted = {
        "responses": response,
        "response_mask": torch.ones(200),
        "teacher_input_ids": torch.cat([1000 + torch.arange(60), response, torch.full((8,), 999)]),
        "teacher_seq_meta": torch.tensor(pack(60, sub_rows), dtype=torch.int64),
    }
    # degenerate stub (un-hinted or padding row): 2-token base, zero mask
    stub = {
        "responses": torch.tensor([11], dtype=torch.long),
        "response_mask": torch.zeros(1),
        "teacher_input_ids": torch.tensor([8, 11], dtype=torch.long),
        "teacher_seq_meta": torch.tensor(DEGENERATE_META, dtype=torch.int64),
    }
    return hinted, stub


def test_branch_keep_positions_predict_the_scored_span():
    hinted, stub = make_rows()
    rows = teacher_branches(
        teacher_input_ids=njt([hinted["teacher_input_ids"], stub["teacher_input_ids"]]),
        teacher_seq_meta=njt([hinted["teacher_seq_meta"], stub["teacher_seq_meta"]]),
        responses=njt([hinted["responses"], stub["responses"]]),
    )
    spans = [branch.span for _, branches in rows for branch in branches]
    # one sub-row per hint, and every row keeps one (dp-collective lockstep); the stub's is 1 token
    assert [span.parent for span in spans] == [0, 0, 1]

    responses = [hinted["responses"], stub["responses"]]
    outputs = []
    for _, branches in rows:
        for branch in branches:
            span = branch.span
            keep = torch.arange(span.body_start, span.body_start + span.end - span.start)
            # position k predicts token k + 1
            assert torch.equal(branch.ids[keep + 1], responses[span.parent][span.start : span.end])
            outputs.append(branch.ids[keep + 1].float())
    grid = scatter_turn_teacher_outputs(outputs, spans, 2, 200)
    assert torch.equal(grid[0, 10:40], torch.arange(10, 40, dtype=torch.float32))
    assert torch.equal(grid[0, 120:200], torch.arange(120, 200, dtype=torch.float32))
    assert grid[0, :10].abs().sum() == 0 and grid[0, 40:120].abs().sum() == 0
    # the stub row only carries its degenerate first-position value (masked in the loss)
    assert grid[1, 0] == 11 and grid[1, 1:].abs().sum() == 0


def test_meta_without_sub_rows_keeps_a_dummy_student_position():
    """Defensive: a meta with no sub-rows still yields one dummy keep position (graph
    connectivity) at the row's prefix boundary."""
    _, stub = make_rows()
    keep = response_keep_positions(
        njt([stub["teacher_input_ids"]]), njt([stub["responses"]]), njt([torch.tensor([0], dtype=torch.int64)])
    )
    assert keep.unbind()[0].tolist() == [0]


def test_student_keep_positions_add_the_outcome_positions():
    """The outcome term's positions join the supervised spans' once each, on the same -1 shift."""
    hinted, stub = make_rows()
    input_ids = njt([torch.arange(250, dtype=torch.long), torch.arange(2, dtype=torch.long)])  # prompt 50 / 1
    responses = njt([hinted["responses"], stub["responses"]])
    meta = njt([hinted["teacher_seq_meta"], stub["teacher_seq_meta"]])
    outcome = [torch.zeros(200), torch.ones(1)]
    outcome[0][[5, 20, 45, 150]] = 1.0  # 20 and 150 are already in a span

    keep = response_keep_positions(input_ids, responses, meta, njt(outcome))
    expected = sorted({50 + p - 1 for p in [*range(10, 40), *range(120, 200), 5, 45]})
    assert sorted(keep.unbind()[0].tolist()) == expected and len(keep.unbind()[0]) == len(expected)
    assert keep.unbind()[1].tolist() == [0]
    plain = response_keep_positions(input_ids, responses, meta, njt([torch.zeros(200), torch.zeros(1)]))
    assert torch.equal(plain.values(), response_keep_positions(input_ids, responses, meta).values())


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
