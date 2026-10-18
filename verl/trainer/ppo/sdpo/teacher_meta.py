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

A hinted sample ships its teacher sub-rows concatenated in one sequence and a flat int64
``teacher_seq_meta`` of ``[n_sub, *fields per sub-row]`` that maps each sub-row's scored span
back onto the sample's response grid. A contrast teacher appends one more sub-row per sub-row
(same span, other context) after them, with the same fields; an anchored contrast appends a third
group (same span, no note at all) after the contrast sub-rows.
"""

from collections.abc import Iterable, Iterator
from typing import NamedTuple, Optional


class SubRow(NamedTuple):
    """One teacher sub-row: ``total_len`` tokens whose last ``body_len`` are the body (the
    response-side tail), with the scored span at ``body[body_start:]`` mapping onto the
    response's ``[start, end)``."""

    total_len: int
    body_len: int
    body_start: int
    start: int
    end: int


FIELDS = len(SubRow._fields)


def pack(
    sub_rows: Iterable[SubRow], contrast_rows: Iterable[SubRow] = (), anchor_rows: Iterable[SubRow] = ()
) -> list[int]:
    """Flatten sub-rows into the wire meta ``[n_sub, *fields...]``, then the contrast sub-rows, then the
    anchor (no-note) sub-rows."""
    sub_rows, contrast_rows, anchor_rows = list(sub_rows), list(contrast_rows), list(anchor_rows)
    assert not contrast_rows or len(contrast_rows) == len(sub_rows), "one contrast sub-row per sub-row"
    assert not anchor_rows or len(anchor_rows) == len(contrast_rows), "one anchor sub-row per contrast sub-row"
    return [len(sub_rows), *(value for sub_row in [*sub_rows, *contrast_rows, *anchor_rows] for value in sub_row)]


def unpack(flat: list[int], sample: Optional[int] = None, contrast: bool = False, anchor: bool = False) -> list[SubRow]:
    """Inverse of :func:`pack`: the sub-rows, or with ``contrast`` the contrast sub-rows, or with ``anchor``
    the anchor sub-rows (none when the sample ships none); ``sample`` only labels the malformed-meta
    assertion."""
    assert not (contrast and anchor), "unpack one group at a time"
    n_sub, entries = flat[0], flat[1:]
    where = "" if sample is None else f" for sample {sample}"
    assert len(entries) in (FIELDS * n_sub, 2 * FIELDS * n_sub, 3 * FIELDS * n_sub), (
        f"teacher_seq_meta malformed{where}: {n_sub} sub-rows, {len(entries)} ints"
    )
    rows = [SubRow(*entries[FIELDS * j : FIELDS * (j + 1)]) for j in range(len(entries) // FIELDS)]
    group = 2 if anchor else 1 if contrast else 0
    return rows[group * n_sub : (group + 1) * n_sub]


# The un-hinted row's stub: one 2-token sub-row whose 1-token body scores response position 0.
DEGENERATE_META = pack([SubRow(total_len=2, body_len=1, body_start=0, start=0, end=1)])


class SubRowSpan(NamedTuple):
    """The scored span of one exploded sub-row, on its parent sample's response grid;
    ``contrast`` marks a contrast teacher's sub-row and ``anchor`` a no-note one."""

    parent: int
    body_start: int
    start: int
    end: int
    contrast: bool = False
    anchor: bool = False


def body_slices(sub_spans: Iterable[SubRowSpan]) -> Iterator[tuple[int, int, slice, slice]]:
    """Per sub-row ``(j, parent, body_slice, response_slice)``: the body positions the teacher
    scored and the response positions they belong to."""
    for j, span in enumerate(sub_spans):
        length = span.end - span.start
        yield j, span.parent, slice(span.body_start, span.body_start + length), slice(span.start, span.end)
