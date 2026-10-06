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

import pytest
import torch

from verl.workers import engine_workers
from verl.workers.config.actor import SelfDistillationConfig
from verl.workers.engine_workers import ActorRolloutRefWorker


@pytest.fixture(autouse=True)
def cpu_device(monkeypatch):
    monkeypatch.setattr(engine_workers, "get_device_name", lambda: "cpu")


class Params(torch.nn.Module):
    def __init__(self, **tensors: torch.Tensor):
        super().__init__()
        for name, tensor in tensors.items():
            self.register_parameter(name, torch.nn.Parameter(tensor.clone()))


class FakeEngine:
    def __init__(self, module: torch.nn.Module):
        self.module = module

    def get_data_parallel_group(self):
        return None


class FakeTrainingWorker:
    def __init__(self, module: torch.nn.Module):
        self.engine = FakeEngine(module)
        self.saved, self.loaded = [], []

    def save_checkpoint(self, *args):
        self.saved.append(args)

    def load_checkpoint(self, *args):
        self.loaded.append(args)


def make_worker(ref: Params, actor: Params, regularization: str = "ema", rate: float = 0.25):
    worker = object.__new__(ActorRolloutRefWorker)
    worker.sdpo_enabled = True
    worker._is_ref = True
    worker.sdpo_config = SelfDistillationConfig(teacher_regularization=regularization, teacher_update_rate=rate)
    worker.ref = FakeTrainingWorker(ref)
    worker.actor = FakeTrainingWorker(actor)
    return worker


def test_ema_moves_the_teacher_toward_the_actor_and_reports_the_gap():
    ref = Params(a=torch.tensor([1.0, 1.0]), b=torch.tensor([2.0]))
    # an actor-only parameter (the ref is built without MTP) is left out, never paired with another one
    actor = Params(a=torch.tensor([3.0, 1.0]), b=torch.tensor([2.0]), mtp=torch.tensor([5.0]))

    distance = make_worker(ref, actor)._update_teacher_ema()

    assert distance == pytest.approx((4.0 / 14.0) ** 0.5)
    torch.testing.assert_close(ref.a.data, torch.tensor([1.5, 1.0]))
    torch.testing.assert_close(ref.b.data, torch.tensor([2.0]))


@pytest.mark.parametrize(
    "actor",
    [Params(a=torch.ones(2)), Params(a=torch.ones(2), b=torch.ones(3))],
    ids=["missing", "reshaped"],
)
def test_ema_refuses_a_teacher_parameter_without_its_actor_counterpart(actor):
    ref = Params(a=torch.ones(2), b=torch.ones(1))

    with pytest.raises(RuntimeError, match="the EMA teacher's parameter b has no matching actor parameter"):
        make_worker(ref, actor)._update_teacher_ema()


@pytest.mark.parametrize(("regularization", "rate"), [("trust_region", 0.25), ("ema", 0.0)])
def test_non_ema_teachers_keep_the_base_weights(regularization, rate):
    ref = Params(a=torch.ones(2))
    worker = make_worker(ref, Params(a=torch.full((2,), 3.0)), regularization, rate)

    assert worker._update_teacher_ema() is None
    assert worker.save_teacher_checkpoint("/ckpt/global_step_2/teacher") is False
    assert worker.ref.saved == []
    torch.testing.assert_close(ref.a.data, torch.ones(2))


def test_the_ema_teacher_is_saved_and_reloaded_next_to_the_actor(tmp_path):
    worker = make_worker(Params(a=torch.ones(2)), Params(a=torch.ones(2)))
    teacher_dir = tmp_path / "global_step_4" / "teacher"

    assert worker.save_teacher_checkpoint(str(teacher_dir), None, 4, 2) is True
    assert worker.ref.saved == [(str(teacher_dir), None, 4, 2)]

    assert worker.load_teacher_checkpoint(str(teacher_dir)) is False
    assert worker.ref.loaded == []

    teacher_dir.mkdir(parents=True)
    (teacher_dir / "fsdp_config.json").write_text("{}")
    assert worker.load_teacher_checkpoint(str(teacher_dir)) is True
    assert worker.ref.loaded == [(str(teacher_dir), None, False)]
