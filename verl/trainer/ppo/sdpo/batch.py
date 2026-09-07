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
"""What a teacher reads from the rollout batch, what it hands back, and the per-row weights
the trainer derives from either."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class TeacherInputs:
    """Per-row lists of one training batch, in row order (a condensed trajectory is one
    session with one row per segment). ``prompts`` is None when the teacher does not need the
    student's prompt tokens; ``seq_scores`` is the sequence-level reward; ``feedback`` is the
    reward's environment feedback per row (None where there is none); ``traj_of_row`` is the
    trajectory key of each row (the session its TransferQueue key belongs to), so a teacher can
    group segments back into trajectories."""

    prompts: Optional[list[torch.Tensor]]
    responses: list[torch.Tensor]
    response_mask: list[torch.Tensor]
    raw_prompts: list[Any]
    uids: list[Any]
    seq_scores: list[float]
    feedback: list[Optional[str]]
    extra_fields: list[dict]
    traj_of_row: list

    def __len__(self) -> int:
        return len(self.responses)


@dataclass
class TeacherBatch:
    """The teacher fields to write back per row (``teacher_input_ids``, ``self_distillation_mask``,
    ``loss_mask`` and, for spliced rows, ``teacher_seq_meta``), the teacher's own metrics, and
    ``weight_scale``, the per-row multiplier the teacher wants on the row's share of its
    trajectory weight (None means 1.0 everywhere)."""

    fields: dict[str, torch.Tensor]
    metrics: dict[str, float] = field(default_factory=dict)
    weight_scale: Optional[list[float]] = None


def trace_weights(
    supervised_per_row: list[float],
    traj_of_row: list,
    weight_scale: Optional[list[float]] = None,
) -> list[float]:
    """Per-row weight for the seq-mean loss, shared by every teacher: a trajectory counts once
    in total, its segments split that weight by how much supervision each carries (an even
    split would over-weight a segment holding one short supervised span).

    ``weight_scale`` lets a teacher up- or down-weight rows relative to each other. It belongs
    here rather than on the token mask because the per-row loss is a token-mean, in which a
    uniform within-row scale cancels. Weights are renormalised to the supervised-row count,
    so a scale re-allocates influence between rows without changing the update's overall
    scale; raw shares would sum to the number of supervised trajectories and shrink the
    update by the average segments-per-trajectory (~0.6x at our condensation rate).
    """
    traj_supervised: dict = defaultdict(float)
    for traj, n_supervised in zip(traj_of_row, supervised_per_row, strict=True):
        traj_supervised[traj] += n_supervised
    if weight_scale is None:
        weight_scale = [1.0] * len(supervised_per_row)
    weights = [
        (n / traj_supervised[traj] if traj_supervised[traj] > 0 else 0.0) * mult
        for traj, n, mult in zip(traj_of_row, supervised_per_row, weight_scale, strict=True)
    ]
    n_supervised_rows = sum(1 for n in supervised_per_row if n > 0)
    total = sum(weights)
    scale = (n_supervised_rows / total) if total > 0 else 1.0
    return [w * scale for w in weights]
