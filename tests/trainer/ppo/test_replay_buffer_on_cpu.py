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
"""The replay buffer's poll merges ``tq.kv_list()`` into it, so a key the trainer cleared can come
back without its ``global_steps``: such an entry belongs to no step, and sampling skips it."""

import threading
from collections import defaultdict
from types import SimpleNamespace

import pytest

from verl.trainer import main_ppo_sync

STEP = {
    "a_0_0": {"global_steps": 3, "status": "success", "seq_len": 8},
    "b_0_0": {"global_steps": 3, "status": "success", "seq_len": 9},
}


def _buffer(monkeypatch):
    """The buffer without its poll thread."""
    monkeypatch.setattr(main_ppo_sync, "KVBatchMeta", SimpleNamespace)
    buffer = object.__new__(main_ppo_sync.ReplayBuffer)
    buffer.partitions = defaultdict(dict)
    buffer.poll_interval = 0.0
    buffer.lock = threading.Lock()
    buffer._stop_event = threading.Event()
    return buffer


def _poll_once(monkeypatch, buffer, listed):
    """One pass of the poll, with ``listed`` as what ``tq.kv_list()`` returns."""

    def kv_list():
        buffer._stop_event.set()
        return listed

    monkeypatch.setattr(main_ppo_sync, "tq", SimpleNamespace(kv_list=kv_list))
    buffer._poll_from_transfer_queue()


@pytest.mark.parametrize("late_tags", [{}, {"status": "success"}], ids=["no tags", "status only"])
def test_a_cleared_key_back_without_global_steps_is_skipped(monkeypatch, late_tags):
    buffer = _buffer(monkeypatch)
    buffer.add("train", {"old_0_0": {"global_steps": 2, "status": "success"}})
    buffer.remove("train", ["old_0_0"])
    _poll_once(monkeypatch, buffer, {"train": {**STEP, "old_0_0": late_tags}})
    assert buffer.partitions["train"]["old_0_0"] == late_tags

    batch = buffer.sample("train", global_steps=3)
    assert batch.keys == list(STEP)
    assert batch.tags == list(STEP.values())
