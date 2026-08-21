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

import pytest
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


# --- one-hot forcing targets --------------------------------------------------------------

from verl.trainer.ppo.sdpo_teacher import call_target_rows
from verl.workers.utils.losses import apply_onehot_call_targets


def test_call_target_rows_map_replace_and_insert_positions():
    resp = torch.tensor([1, 2, 3, 4, 5, 9, 7, 8], dtype=torch.int64)
    hinted = [(1, 4, 8, "h", "call", "T")]
    rows = call_target_rows([(4, 8)], [True], hinted, resp, lambda t: [5, 6, 7, 8], 8)
    assert rows.tolist() == [-1, -1, -1, -1, -1, 6, -1, -1]
    rows = call_target_rows([(0, 3)], [True], [(0, 0, 3, "h", "call", "T")],
                            torch.tensor([1, 2, 4], dtype=torch.int64), lambda t: [1, 2, 3, 4], 3)
    assert rows.tolist() == [-1, -1, 3], "insert boundary maps to the first missing token"


def test_call_target_rows_skip_fallback_and_turn_hints():
    resp = torch.tensor([1, 2, 3], dtype=torch.int64)
    rows = call_target_rows([(0, 3)], [False], [(0, 0, 3, "h", "call", "T")], resp, lambda t: [9], 3)
    assert rows.tolist() == [-1, -1, -1]
    rows = call_target_rows([(0, 3)], [True], [(0, 0, 3, "h", "turn", None)], resp, lambda t: [9], 3)
    assert rows.tolist() == [-1, -1, -1]


def test_onehot_override_hits_only_target_positions():
    resp = torch.tensor([[5, 9, 7]])
    tgt = torch.tensor([[-1, 6, 7]])
    tlp = torch.zeros(1, 3) - 1.0
    topk_lp = torch.zeros(1, 3, 2) - 2.0
    topk_idx = torch.tensor([[[5, 1], [6, 9], [7, 2]]])
    out_lp, out_topk = apply_onehot_call_targets(tlp, resp, tgt, topk_lp, topk_idx)
    assert out_lp[0].tolist() == [-1.0, -30.0, 0.0]
    assert out_topk[0, 1].tolist() == [0.0, -30.0]
    assert out_topk[0, 0].tolist() == [-2.0, -2.0]


def test_onehot_override_keeps_teacher_when_target_not_in_topk():
    resp = torch.tensor([[9]])
    tgt = torch.tensor([[6]])
    tlp = torch.zeros(1, 1) - 1.0
    topk_lp = torch.zeros(1, 1, 2) - 2.0
    topk_idx = torch.tensor([[[9, 3]]])  # 6 absent from student's top-k
    out_lp, out_topk = apply_onehot_call_targets(tlp, resp, tgt, topk_lp, topk_idx)
    assert out_lp[0].tolist() == [-30.0], "per-token logp still one-hot (student token != target)"
    assert out_topk[0, 0].tolist() == [-2.0, -2.0], "topk falls back to the real teacher"


def test_onehot_override_noop_without_targets():
    tlp = torch.zeros(1, 2) - 1.0
    out_lp, out_topk = apply_onehot_call_targets(tlp, torch.tensor([[1, 2]]), torch.tensor([[-1, -1]]), None, None)
    assert torch.equal(out_lp, tlp) and out_topk is None


# --- forced swap (call_target == "forced") -------------------------------------------------

from verl.trainer.ppo.sdpo_teacher import forced_call_swap, forced_target_spans, splice_row

FCLOSE = torch.tensor([96], dtype=torch.int64)
# one turn [0:12): reasoning [0:4), call [4:11) framed by 95/96, turn tail 99
FRESP = torch.tensor([1, 2, 3, 4, 95, 30, 31, 32, 33, 34, 96, 99], dtype=torch.int64)


def _enc(table):
    return lambda t: table[t]


def test_forced_target_spans_modes():
    orig = [95, 30, 31, 32, 96]
    tgt = [95, 30, 41, 42, 32, 96]  # one replace (31->41) that also inserts 42
    assert forced_target_spans(orig, tgt, "first") == [(2, 4)]
    assert forced_target_spans(orig, tgt, "all") == [(2, 4)]
    assert forced_target_spans(orig, tgt, "span") == [(0, 6)]
    two = [95, 40, 31, 33, 96]  # two independent replaces
    assert forced_target_spans(orig, two, "first") == [(1, 2)]
    assert forced_target_spans(orig, two, "all") == [(1, 2), (3, 4)]


def test_forced_target_spans_delete_supervises_boundary():
    # correction removes 31: the target position after the removal is supervised
    assert forced_target_spans([95, 30, 31, 32, 96], [95, 30, 32, 96], "all") == [(2, 3)]
    # removal at the very end has no following target position: nothing to supervise
    assert forced_target_spans([95, 30, 31], [95, 30], "all") == []


def test_forced_swap_same_length_replace():
    table = {"T": [95, 30, 41, 32, 33, 34, 96]}  # rel pos 2 changes
    hinted = [(0, 0, 12, "h", "call", "T")]
    swap = forced_call_swap(FRESP, hinted, _enc(table), CALL_OPEN, FCLOSE, "first")
    assert swap is not None and swap["hint_idx"] == 0
    assert swap["at"] == 4 and swap["removed"] == 7 and swap["inserted"] == 7
    assert swap["response_ids"].tolist() == [1, 2, 3, 4, 95, 30, 41, 32, 33, 34, 96, 99]
    assert swap["mask_spans"] == [(6, 7)]
    assert swap["hinted_turns"] == [(0, 0, 12, "h", "call", "T")], "delta 0 leaves spans alone"


def test_forced_swap_shifts_later_spans():
    resp = torch.cat([FRESP, torch.tensor([50, 90, 91, 92, 60, 61], dtype=torch.int64)])
    table = {"T": [95, 30, 41, 42, 32, 33, 34, 96]}  # delta +1
    hinted = [(0, 0, 12, "h", "call", "T"), (1, 14, 18, "g", "turn", None)]
    swap = forced_call_swap(resp, hinted, _enc(table), CALL_OPEN, FCLOSE, "all")
    assert swap is not None
    assert swap["response_ids"].shape[0] == resp.shape[0] + 1
    assert swap["hinted_turns"][0] == (0, 0, 13, "h", "call", "T"), "own turn end follows the swap"
    assert swap["hinted_turns"][1] == (1, 15, 19, "g", "turn", None), "later turn shifts by delta"
    assert swap["mask_spans"] == [(6, 8)], "insert block supervised as real target tokens"


def test_forced_swap_skips_unswappable_rows():
    enc = _enc({"T": [95, 30, 96]})
    no_target = [(0, 0, 12, "h", "call", None)]
    assert forced_call_swap(FRESP, no_target, enc, CALL_OPEN, FCLOSE, "first") is None
    turn_hint_only = [(0, 0, 12, "h", "turn", "T")]
    assert forced_call_swap(FRESP, turn_hint_only, enc, CALL_OPEN, FCLOSE, "first") is None
    no_open = torch.tensor([1, 2, 3, 4, 30, 31, 96, 99], dtype=torch.int64)
    assert forced_call_swap(no_open, [(0, 0, 8, "h", "call", "T")], enc, CALL_OPEN, FCLOSE, "first") is None
    truncated = FRESP[:9]  # call opening present, closing cut off
    assert forced_call_swap(truncated, [(0, 0, 9, "h", "call", "T")], enc, CALL_OPEN, FCLOSE, "first") is None
    identical = _enc({"T": FRESP[4:11].tolist()})
    assert forced_call_swap(FRESP, [(0, 0, 12, "h", "call", "T")], identical, CALL_OPEN, FCLOSE, "first") is None


def test_forced_swap_picks_the_call_hint_among_turn_hints():
    resp = torch.cat([FRESP, torch.tensor([50, 90, 91, 92, 60, 61], dtype=torch.int64)])
    table = {"T": [95, 30, 41, 32, 33, 34, 96]}
    hinted = [(1, 14, 18, "g", "turn", None), (0, 0, 12, "h", "call", "T")]
    swap = forced_call_swap(resp, hinted, _enc(table), CALL_OPEN, FCLOSE, "first")
    assert swap is not None and swap["hint_idx"] == 1


def test_splice_row_keeps_dtype_and_positions():
    row = torch.tensor([0.5, 1.5, 2.5, 3.5])
    out = splice_row(row, 1, 2, torch.zeros(3))
    assert out.tolist() == [0.5, 0.0, 0.0, 0.0, 3.5] and out.dtype == row.dtype
    mask = torch.tensor([1, 1, 0, 1], dtype=torch.int64)
    out = splice_row(mask, 2, 1, torch.ones(2))
    assert out.tolist() == [1, 1, 1, 1, 1] and out.dtype == torch.int64


def test_forced_swap_then_teacher_build_stays_aligned():
    table = {"T": [95, 30, 41, 42, 32, 33, 34, 96]}  # delta +1
    hinted = [(0, 0, 12, "h", "call", "T")]
    swap = forced_call_swap(FRESP, hinted, _enc(table), CALL_OPEN, FCLOSE, "all")
    new_resp, new_hinted = swap["response_ids"], swap["hinted_turns"]
    prompt = torch.arange(10, dtype=torch.int64)
    hint = torch.tensor([70, 71], dtype=torch.int64)
    seq, meta, fallbacks, spans, placed = build_spliced_teacher_row(
        prompt, new_resp, new_hinted, [hint], 100, HEADER, close_ids=CLOSE, call_open_ids=CALL_OPEN)
    assert fallbacks == 0 and placed == [True]
    assert spans == [(4, 13)], "teacher scores the corrected call span on the new grid"
    expected = torch.cat([prompt, new_resp[:4], CLOSE, hint, HEADER, new_resp[4:13]])
    assert torch.equal(seq, expected)
    for a, b in swap["mask_spans"]:
        assert spans[0][0] <= a < b <= spans[0][1], "supervised spans sit inside the scored span"
    from verl.trainer.ppo.sdpo_teacher import turn_token_mask
    mask = turn_token_mask(new_resp.shape[0], swap["mask_spans"])
    assert mask.tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], "only the changed corrected tokens train"


# --- channel balance (call_loss_weight) ----------------------------------------------------

from verl.trainer.ppo.sdpo_teacher import trace_weights


def test_trace_weights_default_matches_one_per_supervised_row():
    w = trace_weights([10.0, 0.0, 5.0], [("a", "0"), ("a", "0"), ("b", "0")], [False] * 3)
    assert sum(w) == pytest.approx(2.0), "weights renormalise to the supervised-row count"
    assert w[1] == 0.0, "unsupervised rows stay at zero"


def test_trace_weights_split_a_condensed_trajectory_by_supervision():
    # one trajectory, two segments: the 3:1 supervision split sets their relative weight,
    # and the renormalisation restores the supervised-row count (not the trajectory count)
    w = trace_weights([30.0, 10.0], [("a", "0"), ("a", "0")], [False, False])
    assert w[0] / w[1] == pytest.approx(3.0)
    assert sum(w) == pytest.approx(2.0)


def test_call_loss_weight_reallocates_between_channels_without_changing_scale():
    rows = [("a", "0"), ("b", "0"), ("c", "0"), ("d", "0")]
    sup = [4.0, 4.0, 4.0, 4.0]
    is_call = [True, False, False, False]
    base = trace_weights(sup, rows, is_call, 1.0)
    down = trace_weights(sup, rows, is_call, 0.1)
    assert sum(base) == pytest.approx(4.0) and sum(down) == pytest.approx(4.0), "scale preserved"
    assert down[0] < base[0], "the call row loses influence"
    assert down[1] > base[1], "turn rows gain it"
    assert down[0] / down[1] == pytest.approx(0.1), "ratio between channels IS lambda"


def test_call_loss_weight_zero_silences_call_rows_only():
    w = trace_weights([4.0, 4.0], [("a", "0"), ("b", "0")], [True, False], 0.0)
    assert w[0] == 0.0 and w[1] == pytest.approx(2.0)


def test_call_loss_weight_is_inert_when_every_row_is_one_channel():
    rows, sup = [("a", "0"), ("b", "0")], [4.0, 4.0]
    for lam in (0.1, 1.0, 5.0):
        assert trace_weights(sup, rows, [True, True], lam) == pytest.approx([1.0, 1.0])
        assert trace_weights(sup, rows, [False, False], lam) == pytest.approx([1.0, 1.0])


from verl.trainer.ppo.sdpo_teacher import FORCED_LP_MARKER, restore_forced_rollout_lp


def test_forced_rollout_lp_marker_restores_ratio_one():
    """The forced tokens were never sampled: their IS weight must be exp(0), not the policy's
    own probability of the call the correction is trying to teach."""
    old = torch.tensor([-0.1, -8.0, -7.0, -0.2])
    rollout = splice_row(torch.tensor([-0.1, -0.3, -0.2]), 1, 1,
                         torch.full((2,), FORCED_LP_MARKER))
    fixed = restore_forced_rollout_lp(rollout, old)
    assert fixed.tolist() == pytest.approx([-0.1, -8.0, -7.0, -0.2])
    assert torch.exp(old - fixed).tolist() == pytest.approx([1.0] * 4)
    # what the zero fill did instead: the weight IS the policy's probability of the token
    zeroed = splice_row(torch.tensor([-0.1, -0.3, -0.2]), 1, 1, torch.zeros(2))
    assert torch.exp(old - zeroed)[1].item() < 1e-3


def test_forced_rollout_lp_restore_is_inert_without_a_marker():
    rollout = torch.tensor([-0.1, -0.3, -0.2])
    out = restore_forced_rollout_lp(rollout, torch.tensor([-1.0, -2.0, -3.0]))
    assert out is rollout


from verl.trainer.ppo.sdpo_teacher import divergence_spans as _dspans


def test_divergence_never_supervises_the_turn_closer():
    """The student span ends on <|im_end|> tokens the corrected call cannot contain: the
    trailing delete past the target's end is the closer, not a divergence."""
    stu = [10, 20, 30, 96, 99]  # ... </tool_call> <|im_end|>
    tgt = [10, 21, 30, 96]
    assert _dspans(stu, tgt, "all") == [(1, 2)]
    assert _dspans(stu, tgt, "first") == [(1, 2)]
    # a genuine trailing replace is NOT the closer and stays supervised
    assert _dspans([10, 20, 31], [10, 20, 32], "all") == [(2, 3)]


def test_divergence_token_modes_keep_one_position_per_block():
    stu = [1, 2, 3, 4, 5, 6, 7, 99]
    tgt = [1, 8, 9, 4, 5, 60, 7]  # two changed blocks + trailing closer delete
    assert _dspans(stu, tgt, "all") == [(1, 3), (5, 6)]
    assert _dspans(stu, tgt, "all_tokens") == [(1, 2), (5, 6)]
    assert _dspans(stu, tgt, "first_token") == [(1, 2)]
    assert _dspans(stu, tgt, "first") == [(1, 3)]


def test_forced_target_spans_token_modes():
    orig = [95, 30, 31, 32, 96]
    tgt = [95, 40, 41, 32, 96]
    assert forced_target_spans(orig, tgt, "all") == [(1, 3)]
    assert forced_target_spans(orig, tgt, "all_tokens") == [(1, 2)]
    assert forced_target_spans(orig, tgt, "first_token") == [(1, 2)]


from verl.trainer.ppo.sdpo_teacher import char_divergence_spans


def test_char_divergence_is_immune_to_bpe_boundary_drag():
    """The escape case from the lab: the divergent characters sit inside a merged
    punctuation token; the mask must cover that token, never its equal neighbours."""
    tab = {1: "x = ", 7: "()`.\\", 4: "\\n", 9: "bar", 6: "<|im_end|>"}
    dec = lambda ids: "".join(tab[i] for i in ids)  # noqa: E731
    stu = [1, 7, 4, 9, 6]
    tgt = "x = ()`.\nbar"
    assert char_divergence_spans(stu, tgt, dec, "first_token") == [(1, 2)]
    assert char_divergence_spans(stu, tgt, dec, "first") == [(1, 3)]
    assert char_divergence_spans(stu, tgt, dec, "all_tokens") == [(1, 2)]


def test_char_divergence_skips_the_turn_closer_and_identical_text():
    tab = {5: "foo", 9: "bar", 6: "<|im_end|>"}
    dec = lambda ids: "".join(tab[i] for i in ids)  # noqa: E731
    assert char_divergence_spans([5, 6], "foo", dec, "all") == []
    assert char_divergence_spans([5, 9], "foobar", dec, "all") == []


from verl.trainer.ppo.sdpo_teacher import segmented_char_opcodes


def test_segmented_diff_cannot_jump_the_field_boundary():
    """When new_str repeats old_str, an unanchored diff matches the student's old_str
    against the target's new_str and produces giant cross-field blocks; the per-field
    diff must yield only the real per-field edits."""
    body = "import (\\\\n    A,\\\\n    B\\\\n)"
    fixed = "import (\\n    A,\\n    B\\n)"
    stu = '<tool_call>{"a": 1, "old_str": "' + body + '", "new_str": "' + body + ' # x"}</tool_call><|im_end|>'
    tgt = '<tool_call>{"a": 1, "old_str": "' + fixed + '", "new_str": "' + fixed + ' # x"}</tool_call>'
    ops = [o for o in segmented_char_opcodes(stu, tgt) if o[0] not in ("equal", "closer")]
    assert len(ops) == 6 and all(o[0] == "delete" and o[2] - o[1] == 1 for o in ops), ops
    closer = [o for o in segmented_char_opcodes(stu, tgt) if o[0] == "closer"]
    assert len(closer) == 1 and stu[closer[0][1]:closer[0][2]] == "<|im_end|>"


def test_segmented_diff_falls_back_without_call_structure():
    ops = segmented_char_opcodes("plain a text", "plain b text")
    assert [o[0] for o in ops] == ["equal", "replace", "equal"]
