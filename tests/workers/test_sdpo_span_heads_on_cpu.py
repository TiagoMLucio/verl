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
"""The student's span-only head (label log-probs, entropy, the SDPO processor) runs a chunk of positions at a time and
recomputes each chunk in the backward. A 12288-token hinted turn over a 248k vocabulary otherwise keeps tens of GiB of
float32 intermediates alive through the teacher's forward and the backward; chunking must change what is kept, not
what is computed."""

from types import SimpleNamespace

import pytest
import torch

from verl.utils import torch_functional as verl_F
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.utils.losses import _sdpo_logits_processor

POSITIONS, VOCAB, TOPK = 37, 64, 5


def make_engine(chunk):
    eng = object.__new__(FSDPEngineWithLMHead)
    eng.model_config = SimpleNamespace(get=lambda key, default=None: {"chunked_lm_head_size": chunk}.get(key, default))
    eng.compute_entropy_from_logits = verl_F.entropy_from_logits
    return eng


def processor(student_logits, data, logits_keep_idx):
    return _sdpo_logits_processor(student_logits, SimpleNamespace(distillation_topk=TOPK))


def inputs():
    generator = torch.Generator().manual_seed(0)
    logits = (torch.randn(POSITIONS, VOCAB, generator=generator) * 3).requires_grad_()
    labels = torch.randint(0, VOCAB, (POSITIONS,), generator=generator)
    return logits, labels, torch.arange(POSITIONS)


def reference(logits, labels):
    """The heads on the whole span with plain autograd, written out rather than through the engine."""
    log_probs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    entropy = verl_F.entropy_from_logits(logits)
    out = _sdpo_logits_processor(logits.unsqueeze(0), SimpleNamespace(distillation_topk=TOPK))
    return log_probs, entropy, out["topk_logps"].squeeze(0), out["topk_indices"].squeeze(0)


def objective(log_probs, entropy, topk_logps):
    weights = torch.linspace(0.5, 1.5, topk_logps.shape[-1])
    return log_probs.sum() + 0.3 * entropy.sum() + (topk_logps * weights).sum()


@pytest.mark.parametrize("chunk", [1, 5, POSITIONS, 1000])
def test_chunks_give_the_whole_passes_outputs_and_gradient(chunk):
    logits, labels, keep_idx = inputs()
    log_probs, entropy, outputs = make_engine(chunk)._span_heads(logits, labels, keep_idx, None, processor, True, True)
    got_grad = torch.autograd.grad(objective(log_probs, entropy, outputs["topk_logps"]), logits)[0]

    logits_ref, _, _ = inputs()
    want_log_probs, want_entropy, want_topk_logps, want_topk_indices = reference(logits_ref, labels)
    want_grad = torch.autograd.grad(objective(want_log_probs, want_entropy, want_topk_logps), logits_ref)[0]

    torch.testing.assert_close(log_probs, want_log_probs)
    torch.testing.assert_close(entropy, want_entropy)
    torch.testing.assert_close(outputs["topk_logps"], want_topk_logps)
    assert torch.equal(outputs["topk_indices"], want_topk_indices)
    torch.testing.assert_close(got_grad, want_grad)


def test_without_autograd_the_heads_run_in_one_pass():
    logits, labels, keep_idx = inputs()
    calls = []

    def counting(student_logits, data, logits_keep_idx):
        calls.append(student_logits.shape[1])
        return processor(student_logits, data, logits_keep_idx)

    with torch.no_grad():
        _, _, outputs = make_engine(4)._span_heads(logits, labels, keep_idx, None, counting, False, True)
    assert calls == [POSITIONS]
    assert outputs["topk_logps"].shape == (POSITIONS, TOPK)


def saved_bytes(run):
    """Bytes of the tensors autograd keeps for the backward of ``run`` (each storage once)."""
    seen = {}

    def pack(tensor):
        seen[tensor.untyped_storage().data_ptr()] = tensor.untyped_storage().nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        run()
    return sum(seen.values())


def test_chunking_keeps_a_fraction_of_what_autograd_saved_for_the_whole_span():
    logits, labels, keep_idx = inputs()
    whole = saved_bytes(lambda: reference(logits, labels))
    chunked = saved_bytes(lambda: make_engine(4)._span_heads(logits, labels, keep_idx, None, processor, True, True))
    assert chunked < whole / 3, (chunked, whole)
