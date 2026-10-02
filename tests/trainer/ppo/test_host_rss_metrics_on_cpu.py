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
"""Per-step host RSS: the trainer process's own and the largest actor worker's (verl#6468)."""

from types import SimpleNamespace

import psutil
import pytest

from verl.trainer import main_ppo_sync
from verl.workers.engine_workers import ActorRolloutRefWorker

GiB = 1024**3


def test_the_worker_reports_its_own_rss():
    assert ActorRolloutRefWorker.host_rss_bytes(None) == pytest.approx(psutil.Process().memory_info().rss, rel=0.1)


def test_the_step_logs_this_process_and_the_largest_worker():
    trainer = object.__new__(main_ppo_sync.PPOTrainer)
    trainer.actor_rollout_wg = SimpleNamespace(host_rss_bytes=lambda: [3 * GiB, 7 * GiB, 5 * GiB])
    out = trainer._host_rss_metrics()
    assert out["perf/worker_rss_gb_max"] == 7.0
    assert out["perf/trainer_rss_gb"] == pytest.approx(psutil.Process().memory_info().rss / GiB, rel=0.1)
