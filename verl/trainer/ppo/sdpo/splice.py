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
"""The spliced teacher row: one sub-row per hint, its meta packed by
:mod:`verl.trainer.ppo.sdpo.teacher_meta`, plus the per-token mask."""

from typing import Optional

import torch

from verl.trainer.ppo.sdpo.hints import HintedTurn
from verl.trainer.ppo.sdpo.teacher_meta import SubRow, pack

__all__ = ["build_spliced_teacher_row", "turn_token_mask"]


def _find_subseq(haystack: torch.Tensor, needle: torch.Tensor, start: int, end: int) -> Optional[int]:
    """Index of the first occurrence of ``needle`` inside ``haystack[start:end]``, or None."""
    n = needle.shape[0]
    if n == 0 or end - start < n:
        return None
    window = haystack[start:end]
    hits = (window.unfold(0, n, 1) == needle).all(dim=1).nonzero()
    return start + hits[0].item() if len(hits) else None


def _sub_row(
    prefix: torch.Tensor, body: list[torch.Tensor], scored: int, span: tuple[int, int]
) -> tuple[torch.Tensor, SubRow]:
    """Concatenate one sub-row from its prefix and ordered body pieces; ``body[scored]`` is
    the span the teacher scores."""
    body_len = sum(part.shape[0] for part in body)
    body_start = sum(part.shape[0] for part in body[:scored])
    return torch.cat([prefix, *body]), SubRow(prefix.shape[0] + body_len, body_len, body_start, *span)


def build_spliced_teacher_row(
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    hinted_turns: list[HintedTurn],
    hint_ids_list: list[torch.Tensor],
    max_prefix_len: int,
    header_ids: torch.Tensor,
    close_ids: Optional[torch.Tensor] = None,
    call_open_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, list[int], int, list[tuple[int, int]]]:
    """One teacher sub-row per hinted turn, concatenated into a single row.

    Each sub-row is the trajectory up to its own turn with only its own hint spliced in,
    and truncated after the scored span. Turn-placed hints go immediately before the
    turn's assistant header and score the whole turn. Call-placed hints go between
    the turn's reasoning and its tool call: the assistant turn is closed (``close_ids``),
    the hint's user turn inserted, the assistant header reopened, and only the call span
    (from ``call_open_ids``, e.g. the ``<tool_call>`` token, to the turn's end) is scored.
    A call hint whose turn has no call opening falls back to the turn splice.

    Carrying every hint in one sequence would make the teacher score a later turn from a
    state it could not reach: it would see its own earlier advice followed by the student
    ignoring it.

    The prompt is left-truncated to ``max_prefix_len``. Returns the concatenation, the packed
    meta (see :mod:`verl.trainer.ppo.sdpo.teacher_meta`), the number of fallback placements
    and the (start, end) spans for the distillation mask (== the meta spans).
    """
    base_prefix = prompt_ids if prompt_ids.shape[0] <= max_prefix_len else prompt_ids[-max_prefix_len:]
    header = header_ids.shape[0]
    pieces: list[torch.Tensor] = []
    sub_rows: list[SubRow] = []
    fallbacks = 0

    for hint, hint_ids in zip(hinted_turns, hint_ids_list, strict=True):
        start, end = hint.start, hint.end
        hint_ids = hint_ids.to(response_ids.dtype)
        prefix = base_prefix
        call_at = None
        if hint.is_call and close_ids is not None and call_open_ids is not None:
            call_at = _find_subseq(response_ids, call_open_ids.to(response_ids.dtype), start, end)

        if call_at is not None:
            body = [
                response_ids[:call_at],  # history plus this turn's header and reasoning
                close_ids.to(response_ids.dtype),
                hint_ids,
                header_ids.to(response_ids.dtype),
                response_ids[call_at:end],  # the call, the only span the teacher scores
            ]
            row, sub_row = _sub_row(prefix, body, scored=4, span=(call_at, end))
        else:
            # one count per hint, whether the call opening was missing, the header was, or both
            degraded = hint.is_call
            if start >= header and torch.equal(response_ids[start - header : start], header_ids):
                insert_at = start - header
            elif start == 0 and prefix.shape[0] >= header and torch.equal(prefix[-header:], header_ids):
                # first turn: its assistant header is the prompt tail, so the hint joins the prefix
                prefix = torch.cat([prefix[:-header], hint_ids, prefix[-header:]])
                insert_at = None
            else:
                insert_at = start
                degraded = True
            fallbacks += int(degraded)

            if insert_at is None:
                body = [response_ids[:start], response_ids[start:end]]
                row, sub_row = _sub_row(prefix, body, scored=1, span=(start, end))
            else:
                body = [
                    response_ids[:insert_at],  # untouched history, no other hints
                    hint_ids,
                    response_ids[insert_at:start],  # the turn's assistant header
                    response_ids[start:end],  # the span the teacher scores
                ]
                row, sub_row = _sub_row(prefix, body, scored=3, span=(start, end))

        pieces.append(row)
        sub_rows.append(sub_row)

    return torch.cat(pieces), pack(sub_rows), fallbacks, [(sub_row.start, sub_row.end) for sub_row in sub_rows]


def turn_token_mask(response_len: int, spans: list[tuple[int, int]]) -> torch.Tensor:
    """Per-token distillation mask: 1 on the scored spans, 0 elsewhere."""
    mask = torch.zeros(response_len, dtype=torch.float32)
    for start, end in spans:
        mask[start:end] = 1.0
    return mask

