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
"""What a teacher reads from the rollout batch and what it hands back."""

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from verl.trainer.ppo.sdpo.hints import HintedTurn


@dataclass
class TeacherInputs:
    """Per-row lists of one training batch, in row order. ``keys`` are the TransferQueue row
    keys ``{uid}_{session}_{index}`` (a condensed trajectory is one session with one row per
    segment); ``prompts`` is None when the teacher does not need the student's prompt tokens;
    ``seq_scores`` is the sequence-level reward; ``feedback`` is the reward's environment
    feedback per row (None where there is none)."""

    keys: list[str]
    prompts: Optional[list[torch.Tensor]]
    responses: list[torch.Tensor]
    response_mask: list[torch.Tensor]
    raw_prompts: list[Any]
    uids: list[Any]
    seq_scores: list[float]
    feedback: list[Optional[str]]
    extra_fields: list[dict]

    def __len__(self) -> int:
        return len(self.keys)


@dataclass
class TeacherBatch:
    """The teacher fields to write back per row (``teacher_input_ids``, ``self_distillation_mask``,
    ``loss_mask`` and, for spliced rows, ``teacher_seq_meta``), the teacher's own metrics, and
    the hints it spliced per row (None for a teacher without hints)."""

    fields: dict[str, torch.Tensor]
    metrics: dict[str, float] = field(default_factory=dict)
    hinted_per_row: Optional[list[list[HintedTurn]]] = None
