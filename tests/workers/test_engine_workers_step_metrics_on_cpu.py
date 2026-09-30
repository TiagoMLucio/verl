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
"""Step metrics the actor worker reports: a fraction over every dp rank, and '__sum' counts that
add across mini-batches instead of being averaged by the trainer's reduce."""

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from verl.trainer.ppo.core_algos import finalize_ratio_metrics
from verl.utils.metric import reduce_metrics
from verl.workers import engine_workers
from verl.workers.engine_workers import _dp_fraction, _sum_counts_over_mini_batches

COUNTS = {0: (1.0, 10.0), 1: (9.0, 10.0)}


def _rank(rank, world, store, out_dir):
    engine_workers.get_device_name = lambda: "cpu"  # gloo reduces host tensors
    dist.init_process_group("gloo", init_method=f"file://{store}", world_size=world, rank=rank)
    try:
        fraction = _dp_fraction(*COUNTS[rank], dist.group.WORLD)
        (out_dir / f"{rank}.txt").write_text(repr(fraction))
    finally:
        dist.destroy_process_group()


def test_the_kept_fraction_is_over_every_rank(tmp_path):
    """Rank 0 alone kept 1 of 10 tokens; over both ranks it is 10 of 20."""
    mp.spawn(_rank, args=(2, tmp_path / "store", tmp_path), nprocs=2, join=True)
    fractions = [float((tmp_path / f"{rank}.txt").read_text()) for rank in range(2)]
    assert fractions == [pytest.approx(0.5), pytest.approx(0.5)]


def test_without_a_group_the_fraction_is_local(monkeypatch):
    monkeypatch.setattr(engine_workers, "get_device_name", lambda: "cpu")
    assert _dp_fraction(1.0, 10.0, None) == pytest.approx(0.1)
    assert _dp_fraction(0.0, 0.0, None) == 0.0


def test_summed_counts_add_across_mini_batches():
    """Two mini-batches of 30 and 50 supervised tokens: the step has 80, and the ratio the pair
    feeds stays a per-token mean either way."""
    metrics = {
        "self_distillation/supervised_tokens__sum": [30.0, 50.0],
        "self_distillation/gap__sum": [-3.0, -5.0],
        "self_distillation/supervised_token_fraction": [0.2, 0.4],
    }
    _sum_counts_over_mini_batches(metrics, epochs=1)
    assert metrics["self_distillation/supervised_tokens__sum"] == [80.0]
    assert metrics["self_distillation/supervised_token_fraction"] == [0.2, 0.4], "only '__sum' keys"
    out = finalize_ratio_metrics(reduce_metrics(metrics))
    assert out["self_distillation/supervised_tokens"] == 80.0
    assert out["self_distillation/gap_mean"] == pytest.approx(-0.1)


def test_epochs_count_the_step_once():
    metrics = {"self_distillation/supervised_tokens__sum": [30.0, 50.0, 30.0, 50.0]}
    _sum_counts_over_mini_batches(metrics, epochs=2)
    assert metrics["self_distillation/supervised_tokens__sum"] == [80.0]
