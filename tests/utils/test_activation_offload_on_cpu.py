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
"""The async activation offload commits one group per FSDP layer in the forward and undoes one per layer in the
backward. A layer that records no backward (a frozen vision tower fed constants, as the Qwen3.5 forward runs on every
text-only batch) must not be one of its groups, or the counters drift past the window map from the second
micro-batch on. Real FSDP2 on one CPU rank; the streams and pinned memory of a GPU are stood in for."""

import contextlib

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import fully_shard

import verl.utils.activation_offload as activation_offload

HIDDEN = 8


class _Stream:
    def wait_stream(self, other):
        pass


class _CpuDevice:
    Stream = _Stream

    @staticmethod
    @contextlib.contextmanager
    def stream(stream):
        yield

    @staticmethod
    def current_stream():
        return _Stream()


@pytest.fixture(scope="module")
def single_process_group(tmp_path_factory):
    if not dist.is_initialized():
        store_file = tmp_path_factory.mktemp("pg") / "store"
        dist.init_process_group(backend="gloo", init_method=f"file://{store_file}", world_size=1, rank=0)
    yield
    dist.destroy_process_group()


@pytest.fixture
def handlers(monkeypatch):
    """The offload handlers a test builds; GPU streams and pinned memory replaced by their CPU stand-ins."""
    monkeypatch.setattr(activation_offload, "get_torch_device", lambda: _CpuDevice)

    def offload(src_tensor, pin_memory=True):
        return src_tensor.device, src_tensor.detach().clone()

    monkeypatch.setattr(activation_offload.SynchronizedGroupOffloadHandler, "offload", staticmethod(offload))
    built = []

    class Recording(activation_offload.AsyncDoubleBufferGroupOffloadHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(self)

    monkeypatch.setattr(activation_offload, "AsyncDoubleBufferGroupOffloadHandler", Recording)
    return built


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x):
        return torch.tanh(self.lin(x))


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(HIDDEN, 2 * HIDDEN)
        self.down = nn.Linear(2 * HIDDEN, HIDDEN)

    def forward(self, x, scale=1.0):
        return x + scale * self.down(torch.relu(self.up(x)))


class _TowerAndLayers(nn.Module):
    """Qwen3.5's shape: a tower of blocks run on constants every forward, whose output is added at weight zero,
    then the decoder layers."""

    def __init__(self, tower, layers):
        super().__init__()
        self.embed = nn.Embedding(32, HIDDEN)
        self.visual = nn.ModuleList(_Block() for _ in range(tower))
        self.layers = nn.ModuleList(_Layer() for _ in range(layers))
        self.head = nn.Linear(HIDDEN, 32)

    def forward(self, ids):
        hidden = self.embed(ids)
        tower = torch.zeros(4, HIDDEN)
        for block in self.visual:
            tower = block(tower)
        hidden = hidden + 0.0 * tower.mean()
        for layer in self.layers:
            hidden = layer(hidden, scale=0.5)
        return self.head(hidden)


def build(tower, layers):
    torch.manual_seed(0)
    model = _TowerAndLayers(tower, layers)
    model.visual.requires_grad_(False)
    mesh = init_device_mesh("cpu", (1,))
    for module in (*model.visual, *model.layers):
        fully_shard(module, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


def micro_batches(count):
    generator = torch.Generator().manual_seed(1)
    return [torch.randint(0, 32, (2, 5), generator=generator) for _ in range(count)]


def grads(model):
    return {name: p.grad.full_tensor() for name, p in model.named_parameters() if p.grad is not None}


def test_a_frozen_tower_is_no_offload_group(single_process_group, handlers):
    model = build(tower=3, layers=4)
    activation_offload.enable_activation_offloading(model, "fsdp2", enable_ckpt=True)
    (handler,) = handlers
    assert (handler.num_layers, handler.num_offload_group) == (4, 3)


@pytest.mark.parametrize("tower", [3, 0], ids=["frozen tower", "no tower"])
def test_every_micro_batch_returns_the_counters_to_zero_and_trains_the_same(single_process_group, handlers, tower):
    batches = micro_batches(3)

    reference = build(tower=tower, layers=4)
    for ids in batches:
        reference(ids).sum().backward()

    model = build(tower=tower, layers=4)
    activation_offload.enable_activation_offloading(model, "fsdp2", enable_ckpt=True)
    (handler,) = handlers
    assert handler.num_layers == 4
    for ids in batches:
        model(ids).sum().backward()
        assert (handler.current_group, handler.offloaded_group_count) == (0, 0)

    want, got = grads(reference), grads(model)
    assert want.keys() == got.keys() and want
    for name in want:
        torch.testing.assert_close(got[name], want[name], atol=1e-6, rtol=1e-5, msg=lambda m, name=name: f"{name}: {m}")


class _LooksCuda(torch.Tensor):
    """A CPU tensor that claims to be on the GPU, since the registration itself needs one."""

    @property
    def is_cuda(self):
        return True


@pytest.fixture
def registrations(monkeypatch):
    """What the offload pins and unpins, as (what, address, bytes); cudaHostRegister itself needs a GPU."""
    log = []

    def register(tensor):
        log.append(("register", tensor.data_ptr(), tensor.numel() * tensor.element_size()))

    def unregister(tensor):
        log.append(("unregister", tensor.data_ptr(), None))

    monkeypatch.setattr(activation_offload, "_register_host", register)
    monkeypatch.setattr(activation_offload, "_unregister_host", unregister)
    return log


def test_a_gpu_activation_is_pinned_in_place_at_its_exact_size_and_unpinned_on_reload(registrations):
    """The pinned allocator rounds every block up to a power of two and keeps it (a 1.25 GiB activation takes 2 GiB for
    good), and pageable memory takes about half its size in device memory on a GH200: the copy is pinned in place."""
    handler = activation_offload.SynchronizedGroupOffloadHandler
    source = torch.randn(5, 7, dtype=torch.bfloat16)
    state = handler.offload(source.as_subclass(_LooksCuda))
    address = state[1].data_ptr()
    assert registrations == [("register", address, 5 * 7 * 2)]

    torch.testing.assert_close(handler.reload(state), source, rtol=0, atol=0)
    assert registrations == [("register", address, 70), ("unregister", address, None)]


def test_a_cpu_activation_is_not_registered(registrations):
    handler = activation_offload.SynchronizedGroupOffloadHandler
    source = torch.randn(4, HIDDEN, dtype=torch.bfloat16)
    state = handler.offload(source)
    torch.testing.assert_close(handler.reload(state), source, rtol=0, atol=0)
    assert registrations == []


def test_the_driver_gets_the_buffer_whole_and_its_error_is_raised(monkeypatch):
    """A registration that fails would leave the copy pageable and bring the device memory growth back unseen."""
    calls = []

    class Cudart:
        def cudaHostRegister(self, address, size, flags):
            calls.append(("register", address, size, flags))
            return "no memory"

        def cudaHostUnregister(self, address):
            calls.append(("unregister", address))
            return "no memory"

    def check_error(result):
        if result != "ok":
            raise RuntimeError(result)

    monkeypatch.setattr(torch.cuda, "cudart", Cudart)
    monkeypatch.setattr(torch.cuda, "check_error", check_error)
    buffer = torch.empty(3, 5, dtype=torch.bfloat16)
    for call in (activation_offload._register_host, activation_offload._unregister_host):
        with pytest.raises(RuntimeError, match="no memory"):
            call(buffer)
    address = buffer.data_ptr()
    assert calls == [("register", address, 30, 0), ("unregister", address)]
