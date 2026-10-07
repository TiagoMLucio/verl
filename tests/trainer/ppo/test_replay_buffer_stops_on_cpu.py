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
"""A prompt that failed in its agent loop worker, or whose worker died, stops sampling: the step would
otherwise train on a partial batch, or wait forever for prompts nobody runs anymore."""

import threading
from collections import defaultdict
from types import SimpleNamespace

import pytest
import ray

from verl.trainer import main_ppo_sync


def _buffer(monkeypatch):
    """The buffer without its poll thread."""
    monkeypatch.setattr(main_ppo_sync, "KVBatchMeta", SimpleNamespace)
    buffer = object.__new__(main_ppo_sync.ReplayBuffer)
    buffer.partitions = defaultdict(dict)
    buffer.owners = defaultdict(dict)
    buffer.poll_interval = 0.0
    buffer.liveness_interval = 0.0
    buffer.lock = threading.Lock()
    buffer._stop_event = threading.Event()
    return buffer


@ray.remote
class Worker:
    def generate_sequences(self):
        return None


@pytest.fixture(scope="module")
def local_ray():
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_a_failed_prompt_stops_sampling(monkeypatch):
    buffer = _buffer(monkeypatch)
    buffer.add(
        "train",
        {
            "a": {"global_steps": 3, "status": "finished"},
            "a_0_0": {"global_steps": 3, "status": "success"},
            "b": {"global_steps": 3, "status": "failure"},
        },
    )
    with pytest.raises(RuntimeError, match=r"1 prompts failed in the agent loop at global_steps=3 \(first: b\)"):
        buffer.sample("train", global_steps=3)


def test_a_finished_step_still_samples(monkeypatch):
    buffer = _buffer(monkeypatch)
    buffer.add("train", {"a": {"global_steps": 3, "status": "finished"}, "a_0_0": {"global_steps": 3, "status": "success"}})
    assert buffer.sample("train", global_steps=3).keys == ["a_0_0"]


def test_a_dead_worker_stops_sampling(monkeypatch, local_ray):
    buffer = _buffer(monkeypatch)
    alive, dead = Worker.remote(), Worker.remote()
    ray.get([alive.__ray_ready__.remote(), dead.__ray_ready__.remote()])
    ray.kill(dead)
    buffer.add(
        "val",
        {uid: {"global_steps": 12, "status": "running"} for uid in ("a", "b", "c")},
    )
    buffer.assign("val", {"a": alive, "b": dead, "c": dead})
    with pytest.raises(RuntimeError, match="worker died at global_steps=12 with 2 prompts unfinished"):
        buffer.sample("val", global_steps=12)


def test_live_or_unrecorded_workers_let_sampling_wait(monkeypatch, local_ray):
    buffer = _buffer(monkeypatch)
    worker = Worker.remote()
    buffer.add("train", {"a": {"global_steps": 5, "status": "running"}})
    buffer._raise_if_a_worker_died("train", ["a"], 5)
    buffer.assign("train", {"a": worker})
    buffer._raise_if_a_worker_died("train", ["a"], 5)
