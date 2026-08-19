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
"""One teacher sub-row per hinted turn, each with a clean history.

A wrong meta encoding misaligns spans rather than raising, so the tokens each sub-row
carries are checked directly.

Runs under pytest, or as ``python tests/trainer/test_teacher_subrows.py``.
"""

import torch

from verl.trainer.ppo.sdpo_teacher import build_spliced_teacher_row
from verl.workers.utils.sdpo import explode_turn_teacher_rows, response_keep_positions

HEADER = torch.tensor([90, 91])
PROMPT = torch.arange(100, 110)
RESPONSE = torch.arange(0, 40)


def _hint(value, length=3):
    return torch.full((length,), value, dtype=torch.int64)


def _build(hinted, hints):
    return build_spliced_teacher_row(PROMPT, RESPONSE, hinted, hints, max_prefix_len=64, header_ids=HEADER)


def test_one_subrow_per_hint():
    hinted = [(3, 12, 16, "a", "turn"), (5, 24, 28, "b", "turn"), (7, 33, 36, "c", "turn")]
    seq, meta, _, _, _ = _build(hinted, [_hint(900), _hint(901), _hint(902)])
    assert meta[0] == 3
    assert len(meta) == 1 + 5 * 3
    assert seq.shape[0] == sum(meta[1 + 5 * j] for j in range(3))


def test_a_subrow_carries_only_its_own_hint():
    """The whole point: sub-row 2 must not contain hint 1, or the teacher scores turn 2
    from a state where it gave advice that was then ignored."""
    hinted = [(3, 12, 16, "a", "turn"), (5, 24, 28, "b", "turn")]
    seq, meta, _, _, _ = _build(hinted, [_hint(900), _hint(901)])

    offset = 0
    carried = []
    for j in range(meta[0]):
        total_len = meta[1 + 5 * j]
        sub = seq[offset : offset + total_len]
        carried.append({900 in sub.tolist(), 901 in sub.tolist()})
        offset += total_len

    assert carried[0] == {True, False}, "sub-row 1 should carry only hint 900"
    assert carried[1] == {False, True}, "sub-row 2 should carry only hint 901"


def test_span_lands_at_body_start():
    """body_start must index the scored tokens inside the body, or the teacher's outputs
    scatter onto the wrong response positions."""
    hinted = [(3, 12, 16, "a", "turn"), (5, 24, 28, "b", "turn")]
    seq, meta, _, _, _ = _build(hinted, [_hint(900), _hint(901)])

    offset = 0
    for j in range(meta[0]):
        total_len, body_len, body_start, start, end = meta[1 + 5 * j : 6 + 5 * j]
        body = seq[offset : offset + total_len][-body_len:]
        torch.testing.assert_close(body[body_start : body_start + (end - start)], RESPONSE[start:end])
        offset += total_len


def test_history_before_the_hint_is_untouched():
    hinted = [(5, 24, 28, "b", "turn")]
    seq, meta, _, _, _ = _build(hinted, [_hint(901)])
    total_len, body_len, body_start, start, end = meta[1:6]
    body = seq[-body_len:]
    # everything before the hint is the verbatim trajectory
    torch.testing.assert_close(body[: start - HEADER.shape[0]], RESPONSE[: start - HEADER.shape[0]])


def test_exploder_returns_one_row_per_hint():
    hinted = [(3, 12, 16, "a", "turn"), (5, 24, 28, "b", "turn"), (7, 33, 36, "c", "turn")]
    seq, meta, _, _, _ = _build(hinted, [_hint(900), _hint(901), _hint(902)])
    seqs = torch.nested.nested_tensor([seq], layout=torch.jagged)
    metas = torch.nested.nested_tensor([torch.tensor(meta)], layout=torch.jagged)
    responses = torch.nested.nested_tensor([RESPONSE], layout=torch.jagged)
    masks = torch.nested.nested_tensor([torch.ones(RESPONSE.shape[0])], layout=torch.jagged)

    sub_seqs, sub_resps, _, parents, spans = explode_turn_teacher_rows(seqs, metas, responses, masks)
    assert parents == [0, 0, 0]
    assert len(spans) == 3 and all(len(s) == 1 for s in spans)
    assert [s[0][1:] for s in spans] == [(12, 16), (24, 28), (33, 36)]
    for j, sub in enumerate(sub_seqs.unbind()):
        assert sub.shape[0] == meta[1 + 5 * j]


def test_student_keeps_the_union_of_spans():
    """The student scores every hinted span in one pass, so its keep positions are the union
    even though the teacher now splits them."""
    hinted = [(3, 12, 16, "a", "turn"), (5, 24, 28, "b", "turn")]
    _, meta, _, _, _ = _build(hinted, [_hint(900), _hint(901)])
    prompt_len = PROMPT.shape[0]
    input_ids = torch.nested.nested_tensor(
        [torch.cat([PROMPT, RESPONSE])], layout=torch.jagged
    )
    responses = torch.nested.nested_tensor([RESPONSE], layout=torch.jagged)
    metas = torch.nested.nested_tensor([torch.tensor(meta)], layout=torch.jagged)

    keep = response_keep_positions(input_ids, responses, metas).values().tolist()
    want = [prompt_len + p - 1 for p in list(range(12, 16)) + list(range(24, 28))]
    assert keep == want


def test_degenerate_row_encoding_round_trips():
    """The stub an un-hinted row ships must survive the exploder."""
    meta = torch.tensor([1, 2, 1, 0, 0, 1])
    seqs = torch.nested.nested_tensor([torch.tensor([100, 0])], layout=torch.jagged)
    metas = torch.nested.nested_tensor([meta], layout=torch.jagged)
    responses = torch.nested.nested_tensor([RESPONSE], layout=torch.jagged)
    masks = torch.nested.nested_tensor([torch.ones(RESPONSE.shape[0])], layout=torch.jagged)

    sub_seqs, sub_resps, _, parents, spans = explode_turn_teacher_rows(seqs, metas, responses, masks)
    assert parents == [0]
    assert spans == [[(0, 0, 1)]]
    assert sub_resps.unbind()[0].shape[0] == 1


def test_first_turn_hint_joins_the_prefix():
    """A turn starting at position 0 has its header in the prompt, so the hint goes there."""
    prompt = torch.cat([torch.arange(100, 108), HEADER])
    hinted = [(1, 0, 4, "a", "turn")]
    seq, meta, fallbacks, _, _ = build_spliced_teacher_row(
        prompt, RESPONSE, hinted, [_hint(900)], max_prefix_len=64, header_ids=HEADER
    )
    assert fallbacks == 0
    assert 900 in seq.tolist()
    total_len, body_len, body_start, start, end = meta[1:6]
    assert (start, end) == (0, 4)
    torch.testing.assert_close(seq[-body_len:][body_start : body_start + 4], RESPONSE[0:4])


def test_fallback_counted_when_no_header_precedes_the_span():
    hinted = [(3, 13, 17, "a", "turn")]  # 13-2=11,12 are not the header tokens
    _, meta, fallbacks, _, _ = _build(hinted, [_hint(900)])
    assert fallbacks == 1
    assert meta[0] == 1


if __name__ == "__main__":
    failures = 0
    for name, case in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            case()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    raise SystemExit(1 if failures else 0)
