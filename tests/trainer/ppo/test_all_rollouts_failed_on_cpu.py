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
"""A training step stops the run when every rollout it sampled ended in an infra failure."""

from types import SimpleNamespace

import pytest

from verl.trainer import main_ppo_sync


class KV:
    def __init__(self, extra_fields):
        self.extra_fields = extra_fields
        self.calls = 0

    def kv_batch_get(self, keys, partition_id, select_fields=None):
        self.calls += 1
        assert select_fields == ["extra_fields"]
        return {"extra_fields": [self.extra_fields[k] for k in keys]}


def _step(monkeypatch, extra_fields):
    kv = KV(extra_fields)
    monkeypatch.setattr(main_ppo_sync, "tq", kv)
    trainer = object.__new__(main_ppo_sync.PPOTrainer)
    trainer.global_steps = 7
    trainer._abort_if_all_rollouts_failed(SimpleNamespace(keys=list(extra_fields), partition_id="train"))
    return kv


def test_every_row_an_infra_failure_aborts(monkeypatch):
    rows = {
        "a_0_0": {"traj_exit_reason": "setup_timeout"},
        "a_0_1": {"traj_exit_reason": "setup_timeout"},
        "b_0_0": {"traj_exit_reason": "agent_loop_failed"},
        "c_0_0": {"traj_exit_reason": "terminal_dead"},
        "d_0_0": {"traj_exit_reason": "generation_timeout"},
        "d_1_0": {"traj_exit_reason": "episode_timeout"},
        "e_0_0": {"traj_exit_reason": "build_failed"},
    }
    with pytest.raises(RuntimeError, match=r"global_steps=7: .*'setup_timeout': 2"):
        _step(monkeypatch, rows)


@pytest.mark.parametrize("survivor", ["finished", "token_limit", "timeout_budget_exhausted", None])
def test_one_rollout_the_harness_did_not_end_keeps_the_step(monkeypatch, survivor):
    rows = {"a_0_0": {"traj_exit_reason": "terminal_dead"}, "b_0_0": {"traj_exit_reason": survivor}}
    assert _step(monkeypatch, rows).calls == 1


def test_rows_without_extra_fields_keep_the_step(monkeypatch):
    _step(monkeypatch, {"a_0_0": None, "b_0_0": {"traj_exit_reason": "agent_loop_failed"}})


def test_an_empty_batch_aborts_without_reading_the_queue(monkeypatch):
    with pytest.raises(RuntimeError, match="no rollout returned"):
        _step(monkeypatch, {})
