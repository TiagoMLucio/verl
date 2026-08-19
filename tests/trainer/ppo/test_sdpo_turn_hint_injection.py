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
"""User-turn hint insertion in build_spliced_teacher_row: the hint lands before the turn's
assistant header (not after it, prefill-style), with a bare-splice fallback when the header
isn't where expected. Meta must keep mapping each span's verbatim tokens on the body grid."""

import torch

from verl.trainer.ppo.sdpo_teacher import build_spliced_teacher_row

HEADER = torch.tensor([90, 91, 92], dtype=torch.int64)


def _spans_map_back(seq, meta, response_ids, hinted_turns):
    """meta is [n_sub, (total_len, body_len, body_start, start, end) per sub-row]; each
    scored span must reproduce the student's tokens verbatim on the sub-row grid."""
    n_sub, rest = meta[0], meta[1:]
    assert n_sub == len(hinted_turns) and len(rest) == 5 * n_sub
    offset = 0
    for j in range(0, len(rest), 5):
        total, body_len, body_start, start, end = rest[j : j + 5]
        prefix_len = total - body_len
        span = seq[offset + prefix_len + body_start : offset + prefix_len + body_start + (end - start)]
        assert torch.equal(span, response_ids[start:end]), f"span [{start},{end}) corrupted"
        offset += total
    assert offset == seq.shape[0]


def test_hint_inserted_before_assistant_header():
    prompt = torch.arange(10, dtype=torch.int64)
    # response: turn0 [0:4), obs+header [4:12) with header at [9:12), turn1 [12:16)
    response = torch.tensor([0, 1, 2, 3, 50, 51, 52, 53, 54, 90, 91, 92, 10, 11, 12, 13], dtype=torch.int64)
    hinted = [(1, 12, 16, "hint", "turn")]
    hint = torch.tensor([70, 71], dtype=torch.int64)

    seq, meta, fallbacks, _, _ = build_spliced_teacher_row(prompt, response, hinted, [hint], 100, HEADER)

    assert fallbacks == 0
    expected = torch.cat([prompt, response[:9], hint, response[9:16]])
    assert torch.equal(seq, expected), "hint must sit before the assistant header, not after it"
    _spans_map_back(seq, meta, response, hinted)


def test_first_turn_hint_joins_prompt_tail():
    prompt = torch.cat([torch.arange(5, dtype=torch.int64), HEADER])
    response = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    hinted = [(0, 0, 4, "hint", "turn")]
    hint = torch.tensor([70, 71], dtype=torch.int64)

    seq, meta, fallbacks, _, _ = build_spliced_teacher_row(prompt, response, hinted, [hint], 100, HEADER)

    assert fallbacks == 0
    expected = torch.cat([prompt[:-3], hint, HEADER, response])
    assert torch.equal(seq, expected)
    assert meta == [1, seq.shape[0], 4, 0, 0, 4], "hint in the prefix must not count toward the body"
    _spans_map_back(seq, meta, response, hinted)


def test_missing_header_falls_back_to_bare_splice():
    prompt = torch.arange(10, dtype=torch.int64)
    response = torch.tensor([0, 1, 2, 3, 50, 51, 52, 10, 11, 12], dtype=torch.int64)  # no header anywhere
    hinted = [(1, 7, 10, "hint", "turn")]
    hint = torch.tensor([70, 71], dtype=torch.int64)

    seq, meta, fallbacks, _, _ = build_spliced_teacher_row(prompt, response, hinted, [hint], 100, HEADER)

    assert fallbacks == 1
    expected = torch.cat([prompt, response[:7], hint, response[7:10]])
    assert torch.equal(seq, expected)
    _spans_map_back(seq, meta, response, hinted)


def test_cumulative_hints_and_truncation_after_last_span():
    prompt = torch.arange(4, dtype=torch.int64)
    response = torch.cat(
        [
            torch.tensor([0, 1], dtype=torch.int64),  # obs
            HEADER,
            torch.tensor([10, 11, 12], dtype=torch.int64),  # turn a: [5:8)
            torch.tensor([60, 61], dtype=torch.int64),  # obs
            HEADER,
            torch.tensor([20, 21], dtype=torch.int64),  # turn b: [13:15)
            torch.tensor([98, 99], dtype=torch.int64),  # trailing tokens beyond last span
        ]
    )
    hinted = [(0, 5, 8, "a", "turn"), (1, 13, 15, "b", "turn")]
    hints = [torch.tensor([70], dtype=torch.int64), torch.tensor([71], dtype=torch.int64)]

    seq, meta, fallbacks, _, _ = build_spliced_teacher_row(prompt, response, hinted, hints, 100, HEADER)

    assert fallbacks == 0
    expected = torch.cat(
        [prompt, response[:2], hints[0], response[2:10], hints[1], response[10:15]]
    )
    assert torch.equal(seq, expected), "both hints before their headers; sequence cut after last span"
    _spans_map_back(seq, meta, response, hinted)


# --- mid-turn (at == "call") splice -------------------------------------------------------

CLOSE = torch.tensor([80, 81], dtype=torch.int64)
CALL_OPEN = torch.tensor([95], dtype=torch.int64)


def test_call_hint_splices_between_reasoning_and_call():
    prompt = torch.arange(10, dtype=torch.int64)
    # one turn [0:10): reasoning [0:4), call opening at 4, call body [5:10)
    response = torch.tensor([1, 2, 3, 4, 95, 30, 31, 32, 33, 34], dtype=torch.int64)
    hinted = [(0, 0, 10, "h", "call")]
    hint = torch.tensor([70, 71], dtype=torch.int64)

    seq, meta, fallbacks, spans, placed = build_spliced_teacher_row(
        prompt, response, hinted, [hint], 100, HEADER, close_ids=CLOSE, call_open_ids=CALL_OPEN)

    assert fallbacks == 0
    expected = torch.cat([prompt, response[:4], CLOSE, hint, HEADER, response[4:10]])
    assert torch.equal(seq, expected), "close + hint + header must sit between reasoning and call"
    assert spans == [(4, 10)], "only the call tokens are scored"
    assert meta == [1, seq.shape[0], seq.shape[0] - 10, 4 + 2 + 2 + 3, 4, 10]
    from verl.trainer.ppo.sdpo_teacher import turn_token_mask
    mask = turn_token_mask(10, spans)
    assert mask.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]


def test_call_hint_without_call_opening_falls_back_to_turn_splice():
    prompt = torch.arange(10, dtype=torch.int64)
    # header at [2:5), turn [5:9), no CALL_OPEN token anywhere
    response = torch.tensor([1, 2, 90, 91, 92, 10, 11, 12, 13], dtype=torch.int64)
    hinted = [(1, 5, 9, "h", "call")]
    hint = torch.tensor([70, 71], dtype=torch.int64)

    seq, meta, fallbacks, spans, placed = build_spliced_teacher_row(
        prompt, response, hinted, [hint], 100, HEADER, close_ids=CLOSE, call_open_ids=CALL_OPEN)

    assert fallbacks == 1, "a call hint with no call opening is a fallback"
    expected = torch.cat([prompt, response[:2], hint, response[2:9]])
    assert torch.equal(seq, expected), "fallback is the turn splice before the header"
    assert spans == [(5, 9)], "fallback scores the whole turn"


def test_call_hint_without_splice_ids_falls_back():
    prompt = torch.arange(10, dtype=torch.int64)
    response = torch.tensor([1, 2, 90, 91, 92, 10, 11, 95, 13], dtype=torch.int64)
    hinted = [(1, 5, 9, "h", "call")]
    hint = torch.tensor([70, 71], dtype=torch.int64)

    seq, meta, fallbacks, spans, placed = build_spliced_teacher_row(prompt, response, hinted, [hint], 100, HEADER)

    assert fallbacks == 1
    assert spans == [(5, 9)]
    assert torch.equal(seq, torch.cat([prompt, response[:2], hint, response[2:9]]))


def test_select_hinted_turns_reads_call_placement():
    from verl.trainer.ppo.sdpo_teacher import select_hinted_turns

    extra = {"turn_spans": [[0, 0, 4], [1, 4, 9]],
             "turn_feedback": [[0, "a"], [1, "b", "call"], ]}
    assert select_hinted_turns(extra, 9) == [(0, 0, 4, "a", "turn", None), (1, 4, 9, "b", "call", None)]
    extra["turn_feedback"] = [[1, "b", "call", "TARGET"]]
    assert select_hinted_turns(extra, 9) == [(1, 4, 9, "b", "call", "TARGET")]


# --- divergence-window narrowing ----------------------------------------------------------

from verl.trainer.ppo.sdpo_teacher import divergence_spans, narrowed_call_spans, turn_token_mask


def test_divergence_single_replace_first_equals_all():
    student, target = [1, 9, 3, 4], [1, 2, 3, 4]
    assert divergence_spans(student, target, "first") == [(1, 2)]
    assert divergence_spans(student, target, "all") == [(1, 2)]


def test_divergence_two_independent_blocks():
    student, target = [9, 2, 3, 8], [1, 2, 3, 4]
    assert divergence_spans(student, target, "first") == [(0, 1)]
    assert divergence_spans(student, target, "all") == [(0, 1), (3, 4)]


def test_divergence_insert_masks_the_boundary_token():
    # student is MISSING a token: the position that should have produced it is masked
    assert divergence_spans([1, 2, 4], [1, 2, 3, 4], "all") == [(2, 3)]


def test_divergence_delete_masks_the_extra_tokens():
    assert divergence_spans([1, 2, 7, 7, 3], [1, 2, 3], "all") == [(2, 4)]


def test_divergence_realigns_after_a_shifted_block():
    # a wrong prefix, then identical content: only the prefix is masked
    assert divergence_spans([5, 5, 1, 2, 3], [1, 2, 3], "all") == [(0, 2)]


def test_divergence_identical_yields_nothing():
    assert divergence_spans([1, 2, 3], [1, 2, 3], "all") == []


def test_narrowing_applies_only_to_placed_call_hints_with_targets():
    response = torch.tensor([1, 2, 3, 4, 5, 9, 7, 8], dtype=torch.int64)
    hinted = [(0, 0, 4, "h", "turn", None), (1, 4, 8, "h", "call", "T")]
    spans = [(0, 4), (4, 8)]
    encode = lambda t: [5, 6, 7, 8]  # student [5,9,7,8] vs target: token 9 wrong
    out = narrowed_call_spans(spans, [False, True], hinted, response, encode, "first")
    assert out == [(0, 4), (5, 6)], "turn span untouched; call span narrowed to the wrong token"
    mask = turn_token_mask(8, out)
    assert mask.tolist() == [1, 1, 1, 1, 0, 1, 0, 0]


def test_narrowing_skips_fallback_call_hints():
    response = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    hinted = [(0, 0, 4, "h", "call", "T")]
    out = narrowed_call_spans([(0, 4)], [False], hinted, response, lambda t: [9], "all")
    assert out == [(0, 4)], "a call hint that fell back to turn splice keeps the whole span"


def test_narrowing_keeps_span_when_diff_is_empty():
    response = torch.tensor([1, 2, 3], dtype=torch.int64)
    hinted = [(0, 0, 3, "h", "call", "T")]
    out = narrowed_call_spans([(0, 3)], [True], hinted, response, lambda t: [1, 2, 3], "all")
    assert out == [(0, 3)]


def test_span_mode_is_passthrough():
    hinted = [(0, 0, 3, "h", "call", "T")]
    assert narrowed_call_spans([(0, 3)], [True], hinted, torch.tensor([9, 9, 9]), lambda t: [1], "span") == [(0, 3)]
