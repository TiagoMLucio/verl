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
"""Per-step health metrics of an SDPO batch and of a validation pass.

Rows are condensation segments; a trajectory is its ``segment_index == 0`` row and
``traj_of_row`` names the trajectory each row belongs to.
"""

from collections import Counter, defaultdict

import numpy as np

from verl.trainer.ppo.metric_utils import process_validation_metrics

#: The agent loop namespaces under this prefix the per-trajectory metrics it wants reported per
#: step. What they measure is the loop's business, so a new one costs no change here.
AGENT_METRIC_PREFIX = "agent/"

#: Exits the agent did not choose. Reported as one always-emitted fraction: the per-reason keys
#: below only exist once that reason has fired, so a healthy run has nothing to chart or alert
#: on. ``unknown_error`` stays out of it, being what the loop could not classify at all.
HARNESS_ABORT_REASONS = frozenset(
    {"setup_timeout", "agent_loop_failed", "terminal_dead", "timeout_budget_exhausted"}
)


def batch_metrics(
    supervised_per_row: list[float],
    supervised_rows: list[bool],
    response_mask: list,
    traj_of_row: list,
    extra_fields: list[dict],
) -> dict:
    """Row and trajectory counts, and which rows the update can learn from. ``reprompt_sample_fraction``
    is the supervised-row fraction under either teacher; the metric name is kept so runs stay
    comparable."""
    batch_size = len(supervised_per_row)
    n_traces = len(set(traj_of_row))
    segs_per_traj = Counter(traj_of_row)
    sup_segs_per_traj = Counter(traj for traj, n in zip(traj_of_row, supervised_per_row, strict=True) if n > 0)
    unsup_rows = [i for i, n in enumerate(supervised_per_row) if n == 0]
    # decode throughput needs the generated count: response_length also holds the observations
    # fed back to the agent, which are ~3x the tokens the model actually produced
    generated = sum(
        sum(int(span[2]) - int(span[1]) for span in ef.get("turn_spans") or []) for ef in extra_fields
    )
    return {
        "self_distillation/rows_per_step": float(batch_size),
        "self_distillation/traces_per_step": float(n_traces),
        "self_distillation/segments_per_trace_max": float(max(segs_per_traj.values(), default=0)),
        "self_distillation/supervised_segments_per_trace_max": float(max(sup_segs_per_traj.values(), default=0)),
        # rows that run a full student forward+backward for zero gradient
        "self_distillation/unsupervised_row_fraction": len(unsup_rows) / max(batch_size, 1),
        "self_distillation/unsupervised_row_tokens": float(sum(int(response_mask[i].sum()) for i in unsup_rows)),
        "self_distillation/supervised_row_tokens": float(
            sum(int(response_mask[i].sum()) for i in range(batch_size) if supervised_per_row[i] > 0)
        ),
        "self_distillation/reprompt_sample_fraction": sum(supervised_rows) / batch_size,
        "rollout/generated_tokens": float(generated),
        "rollout/generated_tokens_per_trace": generated / n_traces if n_traces else 0.0,
    }


def condensation_metrics(extra_fields: list[dict], seq_scores: list[float], success_threshold: float) -> dict:
    """Condensation reach and whether it predicts the outcome.

    All of a trajectory's segments carry the same reward, so per-trace stats read off the
    first-segment rows.
    """
    traces, turns_by_seg = [], defaultdict(list)
    for i, ef in enumerate(extra_fields):
        seg_idx = int(ef.get("segment_index", 0) or 0)
        spans = ef.get("turn_spans") or []
        if spans:  # failed rollouts ship empty rows; counting them as 0 turns skews the mean
            turns_by_seg[min(seg_idx, 3)].append(len(spans))
        if seg_idx == 0:
            num_segments = max(1, int(ef.get("num_segments", 1) or 1))
            traces.append((num_segments, seq_scores[i], ef.get("traj_exit_reason")))
    if not traces:
        return {}
    n = len(traces)
    solved = lambda score: score >= success_threshold  # noqa: E731
    out = {
        "rollout/condensed_trace_fraction": sum(1 for s, _, _ in traces if s > 1) / n,
        "rollout/segments_per_trace": sum(s for s, _, _ in traces) / n,
        "rollout/harness_abort_fraction": sum(1 for _, _, r in traces if r in HARNESS_ABORT_REASONS) / n,
    }
    for bucket in (1, 2, 3):
        sel = [sc for s, sc, _ in traces if (s == bucket if bucket < 3 else s >= 3)]
        name = f"{bucket}seg" if bucket < 3 else "3plusseg"
        if sel:
            out[f"rollout/solve_rate_{name}"] = sum(1 for sc in sel if solved(sc)) / len(sel)
            out[f"rollout/trace_fraction_{name}"] = len(sel) / n
    reasons = [r for _, _, r in traces if r]
    for reason in sorted(set(reasons)):
        sub = [sc for _, sc, r in traces if r == reason]
        out[f"rollout/exit_{reason}_fraction"] = len(sub) / n
        out[f"rollout/solve_rate_exit_{reason}"] = sum(1 for sc in sub if solved(sc)) / len(sub)
    for seg_idx, counts in sorted(turns_by_seg.items()):
        name = str(seg_idx) if seg_idx < 3 else "3plus"
        out[f"rollout/turns_in_segment_{name}"] = sum(counts) / len(counts)
    return out


def trajectory_timing_metrics(extra_fields: list[dict]) -> dict:
    """Per-trajectory time split. The residual (loop_wall minus the parts) is the in-loop
    overhead we have not attributed yet; step wall clock is set by the slowest trajectory,
    so the max matters more than the mean."""
    rows = [ef.get("timings") or {} for ef in extra_fields if int(ef.get("segment_index", 0) or 0) == 0]
    rows = [t for t in rows if t.get("loop_wall")]
    if not rows:
        return {}
    parts = ("generate_sequences", "tool_calls", "condense", "parse_action", "tokenize_observations")
    out = {}
    for key in parts + ("loop_wall", "env_setup", "reward_eval", "reflect"):
        vals = [float(t.get(key, 0.0)) for t in rows]
        out[f"traj_time/{key}_mean"] = sum(vals) / len(vals)
    residual = [
        max(0.0, float(t.get("loop_wall", 0.0)) - sum(float(t.get(k, 0.0)) for k in parts)) for t in rows
    ]
    totals = [
        float(t.get("loop_wall", 0.0))
        + float(t.get("env_setup", 0.0))
        + float(t.get("reward_eval", 0.0))
        + float(t.get("reflect", 0.0))
        for t in rows
    ]
    # vLLM preemption count per trajectory: >0 means the KV cache could not hold the
    # working set, so sequences were evicted and their prefill recomputed (wasted GPU
    # work that shows up as high utilization with low goodput). -1 = engine did not report.
    preempted = [float(t.get("num_preempted", -1)) for t in rows]
    reported = [p for p in preempted if p >= 0]
    out["rollout/preempted_reported_fraction"] = len(reported) / len(preempted)
    if reported:
        out["rollout/preempted_mean"] = sum(reported) / len(reported)
        out["rollout/preempted_max"] = max(reported)
        out["rollout/preempted_trace_fraction"] = sum(1 for p in reported if p > 0) / len(reported)
    out["traj_time/unattributed_mean"] = sum(residual) / len(residual)
    out["traj_time/total_mean"] = sum(totals) / len(totals)
    out["traj_time/total_max"] = max(totals)
    # the spread between these is the idle tail: the phase lasts as long as the max
    ordered = sorted(totals)
    out["traj_time/total_p50"] = ordered[len(ordered) // 2]
    out["traj_time/total_p90"] = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
    # the slowest trajectory sets the step's wall clock, so its own split is what matters
    slowest = rows[max(range(len(rows)), key=lambda i: totals[i])]
    for key in parts + ("loop_wall", "env_setup", "reward_eval", "reflect"):
        out[f"traj_time/slowest_{key}"] = float(slowest.get(key, 0.0))
    out["traj_time/unattributed_share"] = sum(residual) / max(sum(totals), 1e-6)
    for key in (
        "eval_completed", "patch_apply_failed", "empty_patch", "work_lost", "reflect_failed", "reflect_empty"
    ):
        vals = [float(t[key]) for t in rows if key in t]  # absent means never measured, not OK
        if vals:
            out[f"reward_health/{key}_fraction"] = sum(vals) / len(vals)
    # both readings, since which one a forwarded metric wants is knowledge this side does not have
    for key in sorted({k for t in rows for k in t if k.startswith(AGENT_METRIC_PREFIX)}):
        vals = [float(t[key]) for t in rows if key in t]
        name = key[len(AGENT_METRIC_PREFIX):]
        out[f"agent_loop/{name}_mean"] = sum(vals) / len(vals)
        out[f"agent_loop/{name}_max"] = max(vals)
    capped = [float(t.get("capped_turns", 0.0)) for t in rows]
    out["reward_health/capped_turns_mean"] = sum(capped) / len(capped)
    out["reward_health/capped_rollouts_fraction"] = sum(1.0 for c in capped if c > 0) / len(capped)
    return out


def validation_metrics(data_sources, sample_uids, reward_extra_infos_dict, sample_turns) -> dict[str, float]:
    data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
    # with multiple sources (e.g. difficulty bands) also log the combined total under "all"
    if len(set(data_sources)) > 1:
        merged = process_validation_metrics(["all"] * len(data_sources), sample_uids, reward_extra_infos_dict)
        data_src2var2metric2val.update(merged)
    metric_dict = {}
    for data_source, var2metric2val in data_src2var2metric2val.items():
        core_var = "acc" if "acc" in var2metric2val else "reward"
        for var_name, metric2val in var2metric2val.items():
            n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
            for metric_name, metric_val in metric2val.items():
                if (
                    (var_name == core_var)
                    and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                    and (f"@{n_max}" in metric_name)
                ):
                    metric_sec = "val-core"
                else:
                    metric_sec = "val-aux"
                pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                metric_dict[pfx] = metric_val

    if len(sample_turns) > 0:
        sample_turns = np.array(sample_turns)
        metric_dict["val-aux/num_turns/min"] = sample_turns.min()
        metric_dict["val-aux/num_turns/max"] = sample_turns.max()
        metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

    return metric_dict
