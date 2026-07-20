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
    body_len, triples = meta[0], meta[1:]
    prefix_len = seq.shape[0] - body_len
    for (_, start, end, _), j in zip(hinted_turns, range(0, len(triples), 3), strict=True):
        body_start = triples[j]
        assert triples[j + 1] == start and triples[j + 2] == end
        span = seq[prefix_len + body_start : prefix_len + body_start + (end - start)]
        assert torch.equal(span, response_ids[start:end]), f"span [{start},{end}) corrupted"


def test_hint_inserted_before_assistant_header():
    prompt = torch.arange(10, dtype=torch.int64)
    # response: turn0 [0:4), obs+header [4:12) with header at [9:12), turn1 [12:16)
    response = torch.tensor([0, 1, 2, 3, 50, 51, 52, 53, 54, 90, 91, 92, 10, 11, 12, 13], dtype=torch.int64)
    hinted = [(1, 12, 16, "hint")]
    hint = torch.tensor([70, 71], dtype=torch.int64)

    seq, meta, fallbacks = build_spliced_teacher_row(prompt, response, hinted, [hint], 100, HEADER)

    assert fallbacks == 0
    expected = torch.cat([prompt, response[:9], hint, response[9:16]])
    assert torch.equal(seq, expected), "hint must sit before the assistant header, not after it"
    _spans_map_back(seq, meta, response, hinted)


def test_first_turn_hint_joins_prompt_tail():
    prompt = torch.cat([torch.arange(5, dtype=torch.int64), HEADER])
    response = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    hinted = [(0, 0, 4, "hint")]
    hint = torch.tensor([70, 71], dtype=torch.int64)

    seq, meta, fallbacks = build_spliced_teacher_row(prompt, response, hinted, [hint], 100, HEADER)

    assert fallbacks == 0
    expected = torch.cat([prompt[:-3], hint, HEADER, response])
    assert torch.equal(seq, expected)
    assert meta == [4, 0, 0, 4], "hint in the prefix must not count toward the body"
    _spans_map_back(seq, meta, response, hinted)


def test_missing_header_falls_back_to_bare_splice():
    prompt = torch.arange(10, dtype=torch.int64)
    response = torch.tensor([0, 1, 2, 3, 50, 51, 52, 10, 11, 12], dtype=torch.int64)  # no header anywhere
    hinted = [(1, 7, 10, "hint")]
    hint = torch.tensor([70, 71], dtype=torch.int64)

    seq, meta, fallbacks = build_spliced_teacher_row(prompt, response, hinted, [hint], 100, HEADER)

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
    hinted = [(0, 5, 8, "a"), (1, 13, 15, "b")]
    hints = [torch.tensor([70], dtype=torch.int64), torch.tensor([71], dtype=torch.int64)]

    seq, meta, fallbacks = build_spliced_teacher_row(prompt, response, hinted, hints, 100, HEADER)

    assert fallbacks == 0
    expected = torch.cat(
        [prompt, response[:2], hints[0], response[2:10], hints[1], response[10:15]]
    )
    assert torch.equal(seq, expected), "both hints before their headers; sequence cut after last span"
    _spans_map_back(seq, meta, response, hinted)
