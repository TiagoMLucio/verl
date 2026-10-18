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
"""The old_log_prob pass on the padded (use_remove_padding=False) branch of a Qwen3.5-shaped
model: the fused torch backend (FusedLinearForPPO, 512-token chunks, no (tokens x vocab)
tensor) must return the log-probs and entropy the logits branch returns, and the chunked
entropy flag must run on that branch. Only FSDP wrapping is stubbed; the model, the patched
forward and the engine's input/output preparation are the production ones."""

import contextlib
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

import verl.workers.engine.fsdp.transformer_impl as transformer_impl
from verl.models.transformers.monkey_patch import patch_forward_with_backends
from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

VOCAB = 64
# lengths on either side of the fused kernel's 512-token chunk and of a 2048-token entropy chunk
ROW_LENGTHS = (1100, 700)


def njt(rows):
    return torch.nested.nested_tensor(rows, layout=torch.jagged)


@pytest.fixture(scope="module")
def single_process_group(tmp_path_factory):
    if not dist.is_initialized():
        store_file = tmp_path_factory.mktemp("pg") / "store"
        dist.init_process_group(backend="gloo", init_method=f"file://{store_file}", world_size=1, rank=0)
    yield
    dist.destroy_process_group()


@pytest.fixture
def cpu_ops(monkeypatch):
    monkeypatch.setattr(transformer_impl, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(transformer_impl, "get_device_id", lambda: "cpu")
    import verl.workers.engine.base as engine_base

    monkeypatch.setattr(engine_base, "get_device_name", lambda: "cpu")


def tiny_qwen3_5():
    """A GDN layer and a full-attention layer, fp32, the vision tower the adapter's dummy
    patch forward needs; text-only rows like the trainer's."""
    text = dict(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=False,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=4096,
    )
    vision = dict(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=4,
        temporal_patch_size=2,
        in_channels=3,
        out_hidden_size=32,
        spatial_merge_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    config = Qwen3_5Config(
        text_config=text,
        vision_config=vision,
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return Qwen3_5ForConditionalGeneration(config).float().eval()


@contextlib.contextmanager
def fused_forward(model):
    """The class-level patch the engine installs under model.use_fused_kernels, undone after."""
    original = type(model).forward
    try:
        patch_forward_with_backends(model, use_fused_kernels=True, fused_kernels_backend="torch")
        yield model
    finally:
        type(model).forward = original


def make_engine(module, entropy_chunking=False):
    eng = object.__new__(FSDPEngineWithLMHead)
    eng.module = module
    eng.engine_config = SimpleNamespace(
        ulysses_sequence_parallel_size=1,
        fsdp_size=1,
        forward_only=True,
        entropy_from_logits_with_chunking=entropy_chunking,
        use_torch_compile=False,
        entropy_checkpointing=False,
    )
    eng.model_config = SimpleNamespace(use_fused_kernels=True, get=lambda key, default=None: default)
    eng.ulysses_device_mesh = None
    eng.ulysses_parallel_group = None
    eng.ulysses_sequence_parallel_size = 1
    eng.use_ulysses_sp = False
    eng.use_remove_padding = False
    eng._is_offload_param = False
    eng._is_offload_optimizer = False
    eng._autocast_dtype = torch.float32
    eng.scaler = None
    eng._inference_module = None
    eng.mode = None
    # the engine's __init__ picks the entropy function from the flag
    from verl.utils import torch_functional as verl_F

    eng.compute_entropy_from_logits = (
        verl_F.entropy_from_logits_with_chunking if entropy_chunking else verl_F.entropy_from_logits
    )
    return eng


def make_batch(use_fused_kernels, micro_batch_size_per_gpu, temperature):
    """Two text-only rows shaped like the trainer's: a 4-token prompt and the rest response."""
    torch.manual_seed(1)
    rows = [torch.randint(1, 60, (n,)) for n in ROW_LENGTHS]
    tensor_dict = {
        "input_ids": njt(rows),
        "position_ids": njt([torch.arange(n) for n in ROW_LENGTHS]),
        "response_mask": njt([torch.ones(n - 4, dtype=torch.long) for n in ROW_LENGTHS]),
        "loss_mask": njt([torch.ones(n - 4, dtype=torch.long) for n in ROW_LENGTHS]),
    }
    non_tensor_dict = {
        "use_remove_padding": False,
        "use_fused_kernels": use_fused_kernels,
        "use_dynamic_bsz": False,
        "micro_batch_size_per_gpu": micro_batch_size_per_gpu,
        "calculate_entropy": True,
        "temperature": temperature,
        "pad_token_id": 0,
    }
    return tu.get_tensordict(tensor_dict=tensor_dict, non_tensor_dict=non_tensor_dict)


def old_log_prob(model, use_fused_kernels, micro_batch_size_per_gpu=1, temperature=1.0, entropy_chunking=False):
    engine = make_engine(model, entropy_chunking=entropy_chunking)
    data = make_batch(use_fused_kernels, micro_batch_size_per_gpu, temperature)
    output = engine.forward_backward_batch(data, None, forward_only=True)
    model_output = output["model_output"]
    return list(model_output["log_probs"].unbind()), list(model_output["entropy"].unbind())


def assert_rows_match(fused, unfused, *, drop_last):
    for f, u in zip(fused, unfused, strict=True):
        assert f.shape == u.shape
        # every consumer drops each row's last position (its label is the per-row roll's
        # wrap-around under the fused path and the packed roll's row-crossing token otherwise)
        end = -1 if drop_last else None
        torch.testing.assert_close(f[:end].float(), u[:end].float(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("micro_batch_size_per_gpu", [1, 2], ids=["unpadded", "padded-rows"])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_fused_padded_branch_matches_logits_branch(
    single_process_group, cpu_ops, micro_batch_size_per_gpu, temperature
):
    model = tiny_qwen3_5()
    logits_lp, logits_ent = old_log_prob(model, False, micro_batch_size_per_gpu, temperature)
    with fused_forward(model):
        fused_lp, fused_ent = old_log_prob(model, True, micro_batch_size_per_gpu, temperature)

    assert [t.shape[0] for t in fused_lp] == list(ROW_LENGTHS)
    assert_rows_match(fused_lp, logits_lp, drop_last=True)
    assert_rows_match(fused_ent, logits_ent, drop_last=False)
    assert all(torch.isfinite(t).all() for t in fused_lp + fused_ent)


def test_fused_padded_branch_never_materializes_logits(single_process_group, cpu_ops, monkeypatch):
    """The fused torch backend goes through FusedLinearForPPO in 512-token chunks; the
    logits branch calls the lm_head module on the whole row."""
    from verl.utils.experimental import torch_functional as fused

    chunks = []
    original = fused._fused_linear_for_ppo_fwd

    def spy(hidden_states, *args, **kwargs):
        chunks.append(hidden_states.shape[0])
        return original(hidden_states, *args, **kwargs)

    monkeypatch.setattr(fused, "_fused_linear_for_ppo_fwd", spy)
    model = tiny_qwen3_5()
    head_calls = []
    model.lm_head.register_forward_hook(lambda mod, inp, out: head_calls.append(out.shape))
    with fused_forward(model):
        old_log_prob(model, True)
    assert head_calls == []
    assert sum(chunks) == sum(ROW_LENGTHS) and max(chunks) == 512 and len(chunks) == 3 + 2

    chunks.clear()
    old_log_prob(model, False)
    assert chunks == [] and [s[1] for s in head_calls] == list(ROW_LENGTHS)


@pytest.mark.parametrize("micro_batch_size_per_gpu", [1, 2], ids=["unpadded", "padded-rows"])
def test_entropy_chunking_runs_on_the_padded_logits_branch(single_process_group, cpu_ops, micro_batch_size_per_gpu):
    """fsdp_config.entropy_from_logits_with_chunking splits dim 0, so the padded branch has to
    hand it the (tokens, vocab) layout; the values are the raw call's."""
    model = tiny_qwen3_5()
    raw_lp, raw_ent = old_log_prob(model, False, micro_batch_size_per_gpu)
    chunked_lp, chunked_ent = old_log_prob(model, False, micro_batch_size_per_gpu, entropy_chunking=True)
    assert_rows_match(chunked_lp, raw_lp, drop_last=False)
    assert_rows_match(chunked_ent, raw_ent, drop_last=False)


def span_only_batch(micro_batch_size_per_gpu, temperature):
    """make_batch on the SDPO update's call: span-only logits over one hinted span per row, and every
    position's label log-prob asked for next to them."""
    data = make_batch(False, micro_batch_size_per_gpu, temperature)
    data["logits_keep_positions"] = njt([torch.arange(n - 300, n - 250) for n in ROW_LENGTHS])
    tu.assign_non_tensor(data, all_label_log_probs=True)
    return data


@pytest.mark.parametrize("micro_batch_size_per_gpu", [1, 2], ids=["unpadded", "padded-rows"])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_the_span_only_pass_also_returns_every_positions_label_log_prob(
    single_process_group, cpu_ops, micro_batch_size_per_gpu, temperature
):
    model = tiny_qwen3_5()
    with fused_forward(model):
        fused_lp, _ = old_log_prob(model, True, micro_batch_size_per_gpu, temperature)
        data = span_only_batch(micro_batch_size_per_gpu, temperature)
        model_output = make_engine(model).forward_backward_batch(data, None, forward_only=True)["model_output"]

    every = list(model_output["all_label_log_probs"].unbind())
    assert [t.shape[0] for t in every] == list(ROW_LENGTHS)
    assert_rows_match(every, fused_lp, drop_last=True)
    # the kept positions' own log-probs still come off the span-only logits
    for kept, full, n in zip(model_output["log_probs"].unbind(), fused_lp, ROW_LENGTHS, strict=True):
        torch.testing.assert_close(kept[n - 300 : n - 250], full[n - 300 : n - 250].float(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("micro_batch_size_per_gpu", [1, 2], ids=["unpadded", "padded-rows"])
def test_the_label_log_probs_carry_the_gradient_of_the_logits_they_come_from(
    single_process_group, cpu_ops, micro_batch_size_per_gpu
):
    """Backward through the fused kernel beside the span-only head gives the gradient autograd gives through
    full logits."""
    model = tiny_qwen3_5()

    def summed_label_log_probs(model_output, data, dp_group=None):
        rows = model_output["all_label_log_probs"].unbind()
        return torch.cat([row[:-1] for row in rows]).sum(), {}

    with fused_forward(model):
        engine = make_engine(model)
        engine.engine_config.forward_only = False
        engine.forward_backward_batch(span_only_batch(micro_batch_size_per_gpu, 1.0), summed_label_log_probs)
    fused = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()

    for ids in make_batch(False, 1, 1.0)["input_ids"].unbind():
        log_probs = torch.log_softmax(model(input_ids=ids.unsqueeze(0), use_cache=False).logits[0], -1)
        log_probs[:-1].gather(-1, ids[1:].unsqueeze(-1)).sum().backward()
    reference = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    assert fused.keys() == reference.keys() and "lm_head.weight" in fused
    for name, grad in reference.items():
        torch.testing.assert_close(fused[name], grad, atol=1e-4, rtol=1e-3, msg=name)
