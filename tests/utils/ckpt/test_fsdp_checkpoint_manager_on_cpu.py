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
"""FSDPCheckpointManager.load_checkpoint on a CPU FSDP2 model: every per-rank file is read with
map_location="cpu", and the model, optimizer, lr scheduler and RNG state come back as saved."""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from transformers import LlamaConfig, LlamaForCausalLM

import verl.utils.checkpoint.checkpoint_manager as checkpoint_manager
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager


@pytest.fixture(scope="module")
def single_process_group(tmp_path_factory):
    if not dist.is_initialized():
        store_file = tmp_path_factory.mktemp("pg") / "store"
        dist.init_process_group(backend="gloo", init_method=f"file://{store_file}", world_size=1, rank=0)
    yield
    dist.destroy_process_group()


def test_load_checkpoint_reads_every_file_onto_cpu(single_process_group, tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint_manager, "get_device_name", lambda: "cpu")
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    model = LlamaForCausalLM(config)
    fully_shard(model, mesh=init_device_mesh("cpu", (1,)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (step + 1))
    model(input_ids=torch.randint(0, 64, (1, 4))).logits.sum().backward()
    optimizer.step()
    lr_scheduler.step()

    manager = FSDPCheckpointManager(model, optimizer, lr_scheduler)
    manager.save_checkpoint(str(tmp_path))
    params = {key: value.full_tensor().clone() for key, value in model.state_dict().items()}
    moments = {param: optimizer.state[param]["exp_avg"].full_tensor().clone() for param in model.parameters()}
    last_lr = lr_scheduler.get_last_lr()
    draw = torch.rand(4)

    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
            optimizer.state[param]["exp_avg"].zero_()
    lr_scheduler.step()

    real_load = torch.load
    map_locations = []

    def recording_load(*args, **kwargs):
        map_locations.append(kwargs.get("map_location"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    manager.load_checkpoint(str(tmp_path))

    assert map_locations == ["cpu", "cpu", "cpu"]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value.full_tensor(), params[key])
    for param, moment in moments.items():
        torch.testing.assert_close(optimizer.state[param]["exp_avg"].full_tensor(), moment)
    assert lr_scheduler.get_last_lr() == last_lr
    torch.testing.assert_close(torch.rand(4), draw)
