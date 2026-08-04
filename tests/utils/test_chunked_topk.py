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
"""A hand-written backward stops autograd from checking it, so check it here.

Runs under pytest, or as ``python tests/utils/test_chunked_topk.py`` where pytest
is not installed.
"""

import torch

from verl.utils.experimental.chunked_topk import chunked_gather_logprobs, chunked_topk_logprobs

K = 5
VOCAB = 50
HIDDEN = 16


def _reference(hidden, weight, k, temperature=None):
    """What the un-chunked path computes, with the full logits materialized."""
    logits = (hidden @ weight.t()).float()
    if temperature is not None:
        logits = logits / temperature
    values, indices = logits.topk(k, dim=-1)
    return values - logits.logsumexp(dim=-1, keepdim=True), indices


def test_forward_matches_reference():
    for num_positions, chunk_size in [(7, 512), (16, 4), (33, 8), (512, 512)]:
        torch.manual_seed(0)
        hidden = torch.randn(num_positions, HIDDEN, dtype=torch.float64)
        weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float64)

        got_logps, got_indices = chunked_topk_logprobs(hidden, weight, K, chunk_size=chunk_size)
        want_logps, want_indices = _reference(hidden, weight, K)

        torch.testing.assert_close(got_logps, want_logps)
        assert torch.equal(got_indices, want_indices)


def test_chunking_does_not_change_the_result():
    """Positions are independent at the head, so any partition must agree exactly."""
    torch.manual_seed(1)
    hidden = torch.randn(64, HIDDEN, dtype=torch.float64)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float64)

    baseline, _ = chunked_topk_logprobs(hidden, weight, K, chunk_size=64)
    for chunk_size in (1, 3, 7, 16, 1024):
        other, _ = chunked_topk_logprobs(hidden, weight, K, chunk_size=chunk_size)
        torch.testing.assert_close(other, baseline, rtol=0, atol=0)


def test_gradcheck():
    for chunk_size in (4, 512):
        torch.manual_seed(2)
        hidden = torch.randn(11, HIDDEN, dtype=torch.float64, requires_grad=True)
        weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float64, requires_grad=True)

        def fn(h, w, chunk_size=chunk_size):
            return chunked_topk_logprobs(h, w, K, chunk_size=chunk_size)[0]

        assert torch.autograd.gradcheck(fn, (hidden, weight), eps=1e-6, atol=1e-8)


def test_gradients_match_the_unchunked_path():
    torch.manual_seed(3)
    hidden = torch.randn(23, HIDDEN, dtype=torch.float64)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float64)
    upstream = torch.randn(23, K, dtype=torch.float64)

    chunked_h, chunked_w = hidden.clone().requires_grad_(), weight.clone().requires_grad_()
    chunked_topk_logprobs(chunked_h, chunked_w, K, chunk_size=5)[0].backward(upstream)

    ref_h, ref_w = hidden.clone().requires_grad_(), weight.clone().requires_grad_()
    _reference(ref_h, ref_w, K)[0].backward(upstream)

    torch.testing.assert_close(chunked_h.grad, ref_h.grad)
    torch.testing.assert_close(chunked_w.grad, ref_w.grad)


def test_per_position_temperature():
    torch.manual_seed(4)
    hidden = torch.randn(9, HIDDEN, dtype=torch.float64)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float64)
    temperature = torch.rand(9, 1, dtype=torch.float64) + 0.5
    upstream = torch.randn(9, K, dtype=torch.float64)

    chunked_h, chunked_w = hidden.clone().requires_grad_(), weight.clone().requires_grad_()
    chunked_topk_logprobs(chunked_h, chunked_w, K, temperature=temperature, chunk_size=2)[0].backward(upstream)

    ref_h, ref_w = hidden.clone().requires_grad_(), weight.clone().requires_grad_()
    _reference(ref_h, ref_w, K, temperature=temperature)[0].backward(upstream)

    torch.testing.assert_close(chunked_h.grad, ref_h.grad)
    torch.testing.assert_close(chunked_w.grad, ref_w.grad)


def test_gather_matches_reference():
    """The teacher path: log-probabilities at indices the student chose."""
    torch.manual_seed(5)
    hidden = torch.randn(19, HIDDEN, dtype=torch.float64)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float64)
    indices = torch.randint(0, VOCAB, (19, K))

    got = chunked_gather_logprobs(hidden, weight, indices, chunk_size=6)

    logits = (hidden @ weight.t()).float()
    want = torch.gather(logits, -1, indices) - logits.logsumexp(dim=-1, keepdim=True)
    torch.testing.assert_close(got, want)


def test_peak_memory_is_bounded_by_chunk_size():
    """The point of the exercise: the peak must not track the number of positions."""
    if not torch.cuda.is_available():
        return
    weight = torch.randn(4096, 128, device="cuda", dtype=torch.bfloat16)

    peaks = []
    for num_positions in (512, 4096):
        hidden = torch.randn(num_positions, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        torch.cuda.reset_peak_memory_stats()
        chunked_topk_logprobs(hidden, weight, K, chunk_size=256)[0].sum().backward()
        peaks.append(torch.cuda.max_memory_allocated())

    # 8x the positions must not cost 8x the peak
    assert peaks[1] < peaks[0] * 3, f"peak scaled with positions: {peaks}"


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
