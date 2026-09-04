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
"""The chunked head runs inside the patched forward, where position selection, label
alignment and branch order are easy to get wrong. Every case here compares it against
the eager path through the same model.

Runs under pytest, or as ``python tests/models/test_chunked_lm_head_forward.py``.
"""

import contextlib
import inspect

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from verl.models.transformers.dense_common import forward_with_torch_backend, forward_with_triton_backend
from verl.models.transformers.monkey_patch import patch_forward_with_backends

VOCAB = 64
K = 5


def _tiny_model():
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    return Qwen3ForCausalLM(config).to(torch.float32).eval()


@contextlib.contextmanager
def _patched(model):
    """The patch replaces forward on the class, so put it back afterwards."""
    original = type(model).forward
    try:
        patch_forward_with_backends(model, fused_kernels_backend="torch", use_chunked_lm_head=True)
        yield model
    finally:
        type(model).forward = original


def _eager_reference(model, input_ids, logits_to_keep, temperature=None):
    """topk / label log-probs derived from the materialized logits of the same model."""
    with torch.no_grad():
        hidden = model.model(input_ids=input_ids).last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = model.lm_head(hidden[:, slice_indices, :]).squeeze(0).float()
    if temperature is not None:
        logits = logits / temperature
    labels = torch.roll(input_ids, shifts=-1, dims=-1).squeeze(0)[slice_indices]
    logsumexp = logits.logsumexp(dim=-1, keepdim=True)
    values, indices = logits.topk(K, dim=-1)
    label_logps = (logits.gather(-1, labels.unsqueeze(-1)) - logsumexp).squeeze(-1)
    return values - logsumexp, indices, label_logps


def _run(model, input_ids, logits_to_keep, temperature=None, chunk_size=4):
    with torch.no_grad():
        return model(
            input_ids=input_ids,
            return_dict=True,
            use_cache=False,
            logits_to_keep=logits_to_keep,
            temperature=temperature,
            distill_topk=K,
            distill_chunk_size=chunk_size,
        )


def _assert_matches(model, input_ids, logits_to_keep, temperature=None, chunk_size=4):
    out = _run(model, input_ids, logits_to_keep, temperature, chunk_size)
    want_topk, want_idx, want_label = _eager_reference(model, input_ids, logits_to_keep, temperature)
    torch.testing.assert_close(out.topk_logps.squeeze(0), want_topk, rtol=1e-5, atol=1e-5)
    assert torch.equal(out.topk_indices.squeeze(0), want_idx)
    torch.testing.assert_close(out.log_probs.squeeze(0), want_label, rtol=1e-5, atol=1e-5)


def test_all_positions():
    model = _tiny_model()
    input_ids = torch.randint(0, VOCAB, (1, 24))
    with _patched(model):
        _assert_matches(model, input_ids, logits_to_keep=0)


def test_selected_positions():
    """The case the update actually uses: a scattered subset of hinted positions."""
    model = _tiny_model()
    input_ids = torch.randint(0, VOCAB, (1, 40))
    keep = torch.tensor([1, 2, 3, 17, 18, 30, 31, 32, 33])
    with _patched(model):
        _assert_matches(model, input_ids, logits_to_keep=keep)


def test_labels_follow_the_selection():
    """A label taken from the wrong position is the bug this catches: shuffling which
    positions are kept must change the label log-probs, and still match the reference."""
    model = _tiny_model()
    input_ids = torch.randint(0, VOCAB, (1, 40))
    with _patched(model):
        first = _run(model, input_ids, torch.tensor([2, 5, 9])).log_probs
        second = _run(model, input_ids, torch.tensor([11, 20, 27])).log_probs
        assert not torch.allclose(first, second)
        _assert_matches(model, input_ids, logits_to_keep=torch.tensor([11, 20, 27]))


def test_single_position():
    """An un-hinted row keeps exactly one dummy position."""
    model = _tiny_model()
    input_ids = torch.randint(0, VOCAB, (1, 16))
    with _patched(model):
        _assert_matches(model, input_ids, logits_to_keep=torch.tensor([7]))


def test_chunk_size_boundaries():
    model = _tiny_model()
    input_ids = torch.randint(0, VOCAB, (1, 32))
    keep = torch.arange(0, 13)
    with _patched(model):
        for chunk_size in (1, 3, 13, 64):
            _assert_matches(model, input_ids, logits_to_keep=keep, chunk_size=chunk_size)


def test_scalar_temperature():
    model = _tiny_model()
    input_ids = torch.randint(0, VOCAB, (1, 24))
    with _patched(model):
        _assert_matches(model, input_ids, logits_to_keep=torch.arange(0, 9), temperature=0.7)


def test_eager_path_is_untouched_when_not_chunking():
    """distill_topk=None must still return real logits, since the teacher needs them."""
    model = _tiny_model()
    input_ids = torch.randint(0, VOCAB, (1, 20))
    with _patched(model):
        out = model(
            input_ids=input_ids,
            return_dict=True,
            use_cache=False,
            logits_to_keep=0,
            use_fused_kernels=False,
        )
    assert out.logits is not None
    assert out.logits.shape == (1, 20, VOCAB)


def test_forward_mode_defaults_to_fused():
    """Once the patch is installed every caller must state its mode, because the default
    is the fused path rather than the eager one. The engine forgetting this sent the
    log-prob pass down the fused branch in run 3001809. Asserted on the signature: taking
    the fused branch for real needs Triton on a device.
    """
    for fn in (forward_with_torch_backend, forward_with_triton_backend):
        assert inspect.signature(fn).parameters["use_fused_kernels"].default is True


def test_gradients_flow_to_hidden_and_head():
    model = _tiny_model()
    input_ids = torch.randint(0, VOCAB, (1, 24))
    keep = torch.arange(0, 10)
    with _patched(model):
        out = model(
            input_ids=input_ids,
            return_dict=True,
            use_cache=False,
            logits_to_keep=keep,
            distill_topk=K,
            distill_chunk_size=3,
        )
        (out.topk_logps.sum() + out.log_probs.sum()).backward()

    assert model.lm_head.weight.grad is not None
    assert torch.isfinite(model.lm_head.weight.grad).all()
    assert model.lm_head.weight.grad.abs().sum() > 0
    embed_grad = model.model.embed_tokens.weight.grad
    assert embed_grad is not None and torch.isfinite(embed_grad).all()


def test_gradients_match_the_eager_path():
    """The number that matters: chunking must not change what the model learns."""
    input_ids = torch.randint(0, VOCAB, (1, 28))
    keep = torch.arange(0, 11)

    chunked_model = _tiny_model()
    with _patched(chunked_model):
        out = chunked_model(
            input_ids=input_ids,
            return_dict=True,
            use_cache=False,
            logits_to_keep=keep,
            distill_topk=K,
            distill_chunk_size=3,
        )
        (out.topk_logps.sum() + out.log_probs.sum()).backward()

    eager_model = _tiny_model()
    hidden = eager_model.model(input_ids=input_ids).last_hidden_state
    logits = eager_model.lm_head(hidden[:, keep, :]).squeeze(0).float()
    labels = torch.roll(input_ids, shifts=-1, dims=-1).squeeze(0)[keep]
    logsumexp = logits.logsumexp(dim=-1, keepdim=True)
    values, _ = logits.topk(K, dim=-1)
    label_logps = (logits.gather(-1, labels.unsqueeze(-1)) - logsumexp).squeeze(-1)
    ((values - logsumexp).sum() + label_logps.sum()).backward()

    torch.testing.assert_close(
        chunked_model.lm_head.weight.grad, eager_model.lm_head.weight.grad, rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(
        chunked_model.model.embed_tokens.weight.grad,
        eager_model.model.embed_tokens.weight.grad,
        rtol=1e-4,
        atol=1e-5,
    )


def test_k_larger_than_vocabulary():
    model = _tiny_model()
    input_ids = torch.randint(0, VOCAB, (1, 12))
    with _patched(model), torch.no_grad():
        out = model(
            input_ids=input_ids,
            return_dict=True,
            use_cache=False,
            logits_to_keep=torch.arange(0, 5),
            distill_topk=VOCAB * 4,
            distill_chunk_size=2,
        )
    assert out.topk_logps.shape[-1] == VOCAB


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
