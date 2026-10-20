# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""FSDPEngine.forward_backward_batch logs the split-half gradient measurement of a training pass, read off an FSDP2
module's sharded gradients, and leaves a forward-only pass alone."""

import copy
from itertools import chain
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

import verl.utils.grad_split as grad_split
import verl.workers.engine.fsdp.transformer_impl as impl
from verl.utils import tensordict_utils as tu
from verl.utils.torch_functional import allgather_dict_into_dict

ROWS, DIM = 6, 5


@pytest.fixture
def gloo(monkeypatch, tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/pg", rank=0, world_size=1)
    monkeypatch.setattr(impl, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(grad_split, "get_device_id", lambda: "cpu")
    yield
    dist.destroy_process_group()


def _engine(model):
    def forward_step(micro_batch, loss_function, forward_only):
        loss = ((model(micro_batch["x"]) - micro_batch["y"]) ** 2).sum()
        return loss, {"metrics": {}}

    return SimpleNamespace(
        ulysses_sequence_parallel_size=1,
        module=model,
        get_data_parallel_group=lambda: None,
        get_data_parallel_size=lambda: 1,
        forward_step=forward_step,
    )


def _data():
    gen = torch.Generator().manual_seed(1)
    data = TensorDict(
        {"x": torch.randn(ROWS, DIM, generator=gen), "y": torch.randn(ROWS, 1, generator=gen),
         "loss_mask": torch.ones(ROWS, 1)},
        batch_size=[ROWS],
    )
    tu.assign_non_tensor(data, use_dynamic_bsz=False, micro_batch_size_per_gpu=1)
    return data


def _grad(model, data, rows):
    model.zero_grad()
    ((model(data["x"][rows]) - data["y"][rows]) ** 2).sum().backward()
    return torch.cat([p.grad.flatten() for p in model.parameters()])


def _values(output):
    """Each grad_split metric, logged as a one-item list like a micro-batch metric."""
    lists = {key: val for key, val in output["metrics"].items() if key.startswith("grad_split/")}
    assert lists and all(isinstance(val, list) and len(val) == 1 for val in lists.values())
    return {key: val[0] for key, val in lists.items()}


def _sharded_pass():
    torch.manual_seed(0)
    reference = torch.nn.Sequential(torch.nn.Linear(DIM, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
    model = copy.deepcopy(reference)
    fully_shard(model, mesh=init_device_mesh("cpu", (1,)))
    data = _data()
    output = impl.FSDPEngine.forward_backward_batch(_engine(model), data, loss_function=None, forward_only=False)
    return reference, data, output


def test_a_training_pass_logs_both_halves(gloo):
    reference, data, output = _sharded_pass()

    a, b = _grad(reference, data, slice(0, ROWS // 2)), _grad(reference, data, slice(ROWS // 2, ROWS))
    metrics = _values(output)
    assert metrics["grad_split/rows_a"] == ROWS // 2 and metrics["grad_split/rows_b"] == ROWS - ROWS // 2
    assert metrics["grad_split/sq_a"] == pytest.approx(float(a @ a), rel=1e-5)
    assert metrics["grad_split/sq_b"] == pytest.approx(float(b @ b), rel=1e-5)
    assert metrics["grad_split/dot_ab"] == pytest.approx(float(a @ b), rel=1e-5, abs=1e-6)
    assert metrics["grad_split/cos"] == pytest.approx(float(a @ b) / float((a @ a) * (b @ b)) ** 0.5, rel=1e-4, abs=1e-6)


def test_packed_micro_batches_are_split_into_random_halves(gloo):
    torch.manual_seed(0)
    reference = torch.nn.Sequential(torch.nn.Linear(DIM, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
    model = copy.deepcopy(reference)
    fully_shard(model, mesh=init_device_mesh("cpu", (1,)))
    data = _data()
    lengths = torch.tensor([8, 1, 6, 2, 7, 3])
    data["input_ids"] = torch.zeros(ROWS, 8, dtype=torch.long)
    data["attention_mask"] = (torch.arange(8) < lengths[:, None]).long()
    tu.assign_non_tensor(data, use_dynamic_bsz=True, max_token_len_per_gpu=8)

    output = impl.FSDPEngine.forward_backward_batch(_engine(model), data, loss_function=None, forward_only=False)

    micro, indices = impl.prepare_micro_batches(data=data, dp_group=None, same_micro_num_in_dp=True)
    order = impl.random.Random(1).sample(range(len(micro)), len(micro))
    rows_a = [row for k in order[: len(micro) // 2] for row in indices[k]]
    assert sorted(rows_a) != sorted(row for k in range(len(micro) // 2) for row in indices[k])
    rows_b = [row for row in range(ROWS) if row not in rows_a]
    a, b = _grad(reference, data, rows_a), _grad(reference, data, rows_b)
    metrics = _values(output)
    assert (metrics["grad_split/rows_a"], metrics["grad_split/rows_b"]) == (len(rows_a), len(rows_b))
    assert metrics["grad_split/sq_a"] == pytest.approx(float(a @ a), rel=1e-5)
    assert metrics["grad_split/dot_ab"] == pytest.approx(float(a @ b), rel=1e-5, abs=1e-6)


def test_the_metrics_survive_the_dp_gather_and_the_mini_batch_flatten(gloo):
    _, _, output = _sharded_pass()
    # TrainingWorker._postprocess_output gathers every metric over the dp group, then train_mini_batch flattens it
    gathered = allgather_dict_into_dict(data=output["metrics"], group=dist.group.WORLD)
    flat = {key: list(chain.from_iterable(val)) for key, val in gathered.items() if key.startswith("grad_split/")}
    assert flat["grad_split/cos"] == [_values(output)["grad_split/cos"]]


def test_a_forward_only_pass_logs_nothing(gloo):
    model = torch.nn.Sequential(torch.nn.Linear(DIM, 1))
    fully_shard(model, mesh=init_device_mesh("cpu", (1,)))
    output = impl.FSDPEngine.forward_backward_batch(_engine(model), _data(), loss_function=None, forward_only=True)
    assert not any(key.startswith("grad_split/") for key in output["metrics"])
