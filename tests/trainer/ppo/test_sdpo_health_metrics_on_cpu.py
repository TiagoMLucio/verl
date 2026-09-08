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
"""The health metrics that must read something while a run is healthy.

The per-reason exit keys only exist once that reason has fired, so nothing charts or alerts at
zero. ``harness_abort_fraction`` and ``work_lost_fraction`` are the two that have to be there
before the bad thing happens.
"""

import pytest

from verl.trainer.ppo import sdpo_health_metrics as health


def _traces(reasons):
    """One single-segment, unsolved trajectory per exit reason."""
    fields = [{"segment_index": 0, "num_segments": 1, "turn_spans": [[0, 0, 4]], "traj_exit_reason": r}
              for r in reasons]
    return fields, [0.0] * len(reasons)


def test_the_abort_fraction_is_emitted_at_zero():
    fields, scores = _traces(["finished"] * 8)
    out = health.condensation_metrics(fields, scores, success_threshold=1.0)
    assert out["rollout/harness_abort_fraction"] == 0.0
    # the per-reason keys are what a healthy run does not have
    assert not [k for k in out if k.startswith("rollout/exit_") and "finished" not in k]


@pytest.mark.parametrize("reason", sorted(health.HARNESS_ABORT_REASONS))
def test_every_grouped_reason_counts(reason):
    fields, scores = _traces([reason] + ["finished"] * 3)
    out = health.condensation_metrics(fields, scores, success_threshold=1.0)
    assert out["rollout/harness_abort_fraction"] == 0.25


def test_the_group_is_exits_the_agent_did_not_choose():
    chosen = ["finished", "stuck", "max_step_limit", "token_limit"]
    fields, scores = _traces(chosen)
    assert health.condensation_metrics(fields, scores, 1.0)["rollout/harness_abort_fraction"] == 0.0


def test_unknown_error_is_reported_but_not_grouped():
    """It mixes harness bugs with lost environments, so folding it in would hide both."""
    fields, scores = _traces(["unknown_error", "finished", "finished", "finished"])
    out = health.condensation_metrics(fields, scores, success_threshold=1.0)
    assert out["rollout/harness_abort_fraction"] == 0.0
    assert out["rollout/exit_unknown_error_fraction"] == 0.25


def test_a_batch_with_no_reason_at_all_still_reports_the_fraction():
    fields, scores = _traces([None] * 4)
    assert health.condensation_metrics(fields, scores, 1.0)["rollout/harness_abort_fraction"] == 0.0


def _timing_rows(work_lost):
    return [{"segment_index": 0, "timings": {"loop_wall": 1.0, "empty_patch": float(w), "work_lost": float(w)}}
            for w in work_lost]


def test_work_lost_joins_the_reward_health_family():
    out = health.trajectory_timing_metrics(_timing_rows([1, 0, 0, 0]))
    assert out["reward_health/work_lost_fraction"] == 0.25
    assert out["reward_health/empty_patch_fraction"] == 0.25


def test_work_lost_is_emitted_at_zero():
    out = health.trajectory_timing_metrics(_timing_rows([0, 0]))
    assert out["reward_health/work_lost_fraction"] == 0.0


def test_work_lost_reproduces_the_reference_run():
    """111 of the 8000 rollouts of the reference validation pass applied edits and still
    produced an empty patch, all of them scored as ordinary wrong answers."""
    out = health.trajectory_timing_metrics(_timing_rows([1] * 111 + [0] * 7889))
    assert out["reward_health/work_lost_fraction"] == pytest.approx(0.0138750)


def test_a_run_that_never_measured_it_reports_nothing():
    """Absent means never measured, not healthy: a rollout whose reward never ran must not
    read as one that lost no work."""
    rows = [{"segment_index": 0, "timings": {"loop_wall": 1.0}}]
    assert "reward_health/work_lost_fraction" not in health.trajectory_timing_metrics(rows)
