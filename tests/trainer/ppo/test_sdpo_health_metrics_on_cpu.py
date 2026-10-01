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
zero. ``harness_abort_fraction`` and ``empty_patch_after_source_edit_fraction`` are the two that have to be there
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


def _timing_rows(lost):
    return [{"segment_index": 0, "timings": {"loop_wall": 1.0, "empty_patch": float(w), "empty_patch_after_source_edit": float(w)}}
            for w in lost]


def test_empty_patch_after_source_edit_joins_the_reward_health_family():
    out = health.trajectory_timing_metrics(_timing_rows([1, 0, 0, 0]))
    assert out["reward_health/empty_patch_after_source_edit_fraction"] == 0.25
    assert out["reward_health/empty_patch_fraction"] == 0.25


def test_empty_patch_after_source_edit_is_emitted_at_zero():
    out = health.trajectory_timing_metrics(_timing_rows([0, 0]))
    assert out["reward_health/empty_patch_after_source_edit_fraction"] == 0.0


def test_empty_patch_after_source_edit_reproduces_the_reference_run():
    """111 of the 8000 rollouts of the reference validation pass applied edits and still
    produced an empty patch, all of them scored as ordinary wrong answers (counted under the
    old applied-edits definition; the fraction arithmetic is what this pins)."""
    out = health.trajectory_timing_metrics(_timing_rows([1] * 111 + [0] * 7889))
    assert out["reward_health/empty_patch_after_source_edit_fraction"] == pytest.approx(0.0138750)


def test_a_run_that_never_measured_it_reports_nothing():
    """Absent means never measured, not healthy: a rollout whose reward never ran must not
    read as one that lost no work."""
    rows = [{"segment_index": 0, "timings": {"loop_wall": 1.0}}]
    assert "reward_health/empty_patch_after_source_edit_fraction" not in health.trajectory_timing_metrics(rows)


def test_a_sandbox_that_never_came_up_counts_in_the_setup_rate_not_the_timings():
    ran = [{"segment_index": 0, "timings": {"loop_wall": 2.0, "agent/setup_attempts": float(a),
                                            "agent/setup_retried": float(a > 1)}} for a in (1, 2)]
    dead = [{"segment_index": 0, "timings": {"agent/setup_attempts": 3.0, "agent/setup_retried": 1.0}}]
    out = health.trajectory_timing_metrics(ran + dead)
    assert out["agent_loop/setup_retried_mean"] == pytest.approx(2 / 3)
    assert out["agent_loop/setup_attempts_max"] == 3.0
    assert out["traj_time/loop_wall_mean"] == 2.0
    # no generate call reported a preemption count, so nothing reads as an engine that did not report
    assert "rollout/preempted_reported_fraction" not in out
    only_dead = health.trajectory_timing_metrics(dead)
    assert only_dead == {"agent_loop/setup_attempts_mean": 3.0, "agent_loop/setup_attempts_max": 3.0,
                         "agent_loop/setup_retried_mean": 1.0, "agent_loop/setup_retried_max": 1.0}


def _val(bands):
    uids = [f"u{i // 2}" for i in range(len(bands))]  # two draws per task
    rewards = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0][: len(bands)]
    return ["v128"] * len(bands), uids, {"reward": rewards}, [3] * len(bands), bands


def test_a_band_splits_its_source_next_to_the_total():
    out = health.validation_metrics(*_val(["mid", "mid", "likely", "likely", "rare", "rare"]))
    assert out["val-core/v128/reward/mean@2"] == pytest.approx(0.5)
    assert out["val-core/v128_mid/reward/mean@2"] == 0.5
    assert out["val-core/v128_likely/reward/mean@2"] == 1.0
    assert out["val-core/v128_rare/reward/mean@2"] == 0.0
    assert "val-core/all/reward/mean@2" not in out


def test_val_exit_reasons_are_fractions_of_the_samples_that_report_one():
    sources, uids, infos, turns, bands = _val([None] * 4)
    out = health.validation_metrics(sources, uids, infos, turns, bands,
                                    sample_exit_reasons=["finished", "finished", "terminal_dead", None])
    assert out["val-aux/exit_finished_fraction"] == pytest.approx(2 / 3)
    assert out["val-aux/exit_terminal_dead_fraction"] == pytest.approx(1 / 3)
    assert out["val-aux/harness_abort_fraction"] == pytest.approx(1 / 3)
    assert "val-aux/exit_stuck_fraction" not in out


def test_rows_without_a_band_log_only_their_source():
    out = health.validation_metrics(*_val([None] * 4))
    assert not any("v128_" in key for key in out)
    assert out["val-core/v128/reward/mean@2"] == 0.75
