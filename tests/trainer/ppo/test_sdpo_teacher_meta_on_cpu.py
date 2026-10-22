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

"""The teacher meta with several contrast groups: a single contrast and its anchor keep their layout, the anchor
is the last group with no notes, and several contrast teachers on one span give the log of their mean
probability."""

import pytest
import torch

from verl.trainer.ppo.sdpo.teacher_meta import DEGENERATE_META, SubRow, SubRowSpan, pack, unpack, unpack_row
from verl.workers.utils.sdpo import scatter_turn_teacher_outputs

HINTS = [SubRow(5, 3, 0, 4), SubRow(9, 3, 6, 10)]


def group(note_len: int) -> list[SubRow]:
    return [SubRow(row.split, note_len, row.start, row.end) for row in HINTS]


def test_one_contrast_and_its_anchor_keep_their_layout():
    meta = pack(2, HINTS, group(4), group(0))
    assert meta[:2] == [2, 2] and len(meta) == 2 + 4 * 6
    assert unpack(meta) == HINTS
    assert unpack(meta, contrast=True) == group(4) and unpack(meta, anchor=True) == group(0)
    assert unpack(pack(2, HINTS, group(4)), anchor=True) == []


def test_several_contrast_groups_then_the_anchor():
    meta = pack(2, HINTS, group(4), group(5), group(6), group(0))
    assert unpack(meta, contrast=True) == group(4) + group(5) + group(6)
    assert unpack(meta, anchor=True) == group(0)
    no_anchor = pack(2, HINTS, group(4), group(5))
    assert unpack(no_anchor, contrast=True) == group(4) + group(5) and unpack(no_anchor, anchor=True) == []


def test_an_anchor_needs_a_contrast_and_comes_last():
    with pytest.raises(AssertionError, match="anchor group follows a contrast"):
        pack(2, HINTS, group(0))
    with pytest.raises(AssertionError, match="only the last group"):
        pack(2, HINTS, group(4), group(0), group(5))


def test_unpack_row_slices_every_groups_notes_in_wire_order():
    meta = pack(2, HINTS, group(4), group(5), group(0))
    base = torch.arange(100, 120)
    lengths = [3, 3, 4, 4, 5, 5]
    notes = [torch.full((n,), k) for k, n in enumerate(lengths)]
    got_base, offset, noted = unpack_row(torch.cat([base, *notes]), meta)
    assert torch.equal(got_base, base) and offset == 2
    assert [n.contrast for n in noted] == [False] * 2 + [True] * 4 + [False] * 2
    assert [n.anchor for n in noted] == [False] * 6 + [True] * 2
    for sub, note in zip(noted, notes, strict=False):
        assert torch.equal(sub.note, note)
    assert all(sub.note.numel() == 0 for sub in noted[6:])


def test_the_degenerate_meta_round_trips():
    assert unpack(DEGENERATE_META) == [SubRow(1, 0, 0, 1)]
    assert unpack(DEGENERATE_META, contrast=True) == [] and unpack(DEGENERATE_META, anchor=True) == []


def test_contrast_teachers_on_one_span_give_the_log_of_their_mean_probability():
    torch.manual_seed(0)
    a, b = torch.log_softmax(torch.randn(4, 3), -1), torch.log_softmax(torch.randn(4, 3), -1)
    spans = [SubRowSpan(0, 0, 2, 6, contrast=True), SubRowSpan(0, 0, 2, 6, contrast=True)]
    grid = scatter_turn_teacher_outputs([a, b], spans, 1, 8, contrast=True)
    torch.testing.assert_close(grid[0, 2:6], torch.log((a.exp() + b.exp()) / 2))
    assert grid[0, :2].abs().sum() == 0 and grid[0, 6:].abs().sum() == 0


def test_one_contrast_teacher_scatters_unchanged():
    a = torch.randn(4, 3)
    grid = scatter_turn_teacher_outputs([a], [SubRowSpan(0, 0, 2, 6, contrast=True)], 1, 8, contrast=True)
    assert torch.equal(grid[0, 2:6], a)
