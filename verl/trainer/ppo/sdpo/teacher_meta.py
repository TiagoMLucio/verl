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
"""Wire codec of the per-row SDPO teacher meta.

A hinted sample ships its trajectory once, the base (the student's prompt, left-truncated, then its
response up to the last scored span), followed by every sub-row's note, and a flat int64
``teacher_seq_meta`` of ``[n_sub, response_offset, *fields per sub-row]``: the response starts at
``response_offset`` in the base, and a sub-row is the base up to its ``split``, its own note, then the
base again up to the end of its scored span, which maps onto the response's ``[start, end)``. Each
contrast teacher appends one more group of sub-rows (same splits and spans, its own notes) after them;
an anchored contrast appends a last group (same splits and spans, no notes) after the contrast groups,
which is how it is told apart: a contrast note is never empty.
The sub-rows of a sample differ only from their split on, so the teacher reads the base once.
"""

from collections.abc import Iterable
from typing import NamedTuple, Optional

import torch


class SubRow(NamedTuple):
    """One teacher sub-row: the base up to ``split``, ``note_len`` note tokens, then the base up to the
    scored span's end; the span maps onto the response's ``[start, end)``."""

    split: int
    note_len: int
    start: int
    end: int


FIELDS = len(SubRow._fields)


def pack(response_offset: int, sub_rows: Iterable[SubRow], *groups: Iterable[SubRow]) -> list[int]:
    """Flatten sub-rows into the wire meta ``[n_sub, response_offset, *fields...]``, then each further group
    in order (an empty one is no group): the contrast groups, then the anchor (no-note) group."""
    sub_rows, groups = list(sub_rows), [group for group in map(list, groups) if group]
    assert all(len(group) == len(sub_rows) for group in groups), "every group has one sub-row per sub-row"
    assert all(row.note_len > 0 for group in groups[:-1] for row in group), "only the last group may have no notes"
    assert not groups or groups[0][0].note_len > 0, "an anchor group follows a contrast group"
    rows = [*sub_rows, *(sub_row for group in groups for sub_row in group)]
    return [len(sub_rows), response_offset, *(value for sub_row in rows for value in sub_row)]


def _groups(flat: list[int], sample: Optional[int]) -> tuple[list[list[SubRow]], bool]:
    n_sub, entries = flat[0], flat[2:]
    if n_sub == 0:
        return [[]], False
    where = "" if sample is None else f" for sample {sample}"
    assert entries and len(entries) % (FIELDS * n_sub) == 0, (
        f"teacher_seq_meta malformed{where}: {n_sub} sub-rows, {len(entries)} ints"
    )
    rows = [SubRow(*entries[FIELDS * j : FIELDS * (j + 1)]) for j in range(len(entries) // FIELDS)]
    groups = [rows[at : at + n_sub] for at in range(0, len(rows), n_sub)]
    return groups, len(groups) > 2 and all(row.note_len == 0 for row in groups[-1])


def unpack(flat: list[int], sample: Optional[int] = None, contrast: bool = False, anchor: bool = False) -> list[SubRow]:
    """Inverse of :func:`pack`: the sub-rows, or with ``contrast`` every contrast group's sub-rows in order,
    or with ``anchor`` the anchor sub-rows (none when the sample ships none); ``sample`` only labels the
    malformed-meta assertion."""
    assert not (contrast and anchor), "unpack one group at a time"
    groups, has_anchor = _groups(flat, sample)
    if anchor:
        return groups[-1] if has_anchor else []
    if contrast:
        return [sub_row for group in groups[1 : len(groups) - has_anchor] for sub_row in group]
    return groups[0]


class NotedSubRow(NamedTuple):
    """A shipped sub-row with its note's tokens; ``contrast`` marks a contrast teacher's sub-row and
    ``anchor`` a no-note one."""

    row: SubRow
    note: torch.Tensor
    contrast: bool = False
    anchor: bool = False


def unpack_row(
    seq: torch.Tensor, flat: list[int], sample: Optional[int] = None
) -> tuple[torch.Tensor, int, list[NotedSubRow]]:
    """A shipped teacher row as ``(base, response_offset, sub-rows with their notes)``, in wire order."""
    groups = [
        (sub_row, contrast, anchor)
        for contrast, anchor in ((False, False), (True, False), (False, True))
        for sub_row in unpack(flat, sample, contrast=contrast, anchor=anchor)
    ]
    base_len = seq.shape[0] - sum(sub_row.note_len for sub_row, _, _ in groups)
    noted, at = [], base_len
    for sub_row, contrast, anchor in groups:
        noted.append(NotedSubRow(sub_row, seq[at : at + sub_row.note_len], contrast, anchor))
        at += sub_row.note_len
    return seq[:base_len], flat[1], noted


def sub_row_ids(base: torch.Tensor, response_offset: int, noted: NotedSubRow) -> torch.Tensor:
    """The whole sequence a sub-row stands for: the base up to its split, its note, then the base up to
    the end of its scored span."""
    split, stop = noted.row.split, response_offset + noted.row.end
    return torch.cat([base[:split], noted.note, base[split:stop]])


# The un-hinted row's stub: a 2-token base (the prompt's last token, the response's first) and one sub-row
# with no note scoring response position 0.
DEGENERATE_META = pack(1, [SubRow(split=1, note_len=0, start=0, end=1)])


class SubRowSpan(NamedTuple):
    """The scored span of one sub-row, on its parent sample's response grid; ``body_start`` is where the
    span starts in the sub-row's branch after its first token. ``contrast`` marks a contrast teacher's
    sub-row and ``anchor`` a no-note one."""

    parent: int
    body_start: int
    start: int
    end: int
    contrast: bool = False
    anchor: bool = False
