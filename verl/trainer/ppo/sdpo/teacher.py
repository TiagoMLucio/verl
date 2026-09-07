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
"""The teacher contract: what the trainer hands every teacher, what it reads back, and how
``self_distillation.teacher`` names one."""

from abc import ABC, abstractmethod
from typing import Optional

import hydra
from hydra.errors import InstantiationException

from verl.trainer.ppo.sdpo.batch import TeacherBatch, TeacherInputs


class SDPOTeacher(ABC):
    """Base of every teacher the trainer can run.

    The trainer provides the tokenizer, ``max_prefix_len`` (the student's prompt budget),
    the dataset's ``apply_chat_template_kwargs`` and ``success_reward_threshold``; a teacher's
    own options are keyword-only parameters of its subclass, so an unknown yaml key fails at
    construction with a TypeError naming it. The trainer reads ``needs_prompts`` (whether
    :class:`TeacherInputs` must carry the student's prompt tokens).
    """

    needs_prompts = False

    def __init__(
        self,
        tokenizer,
        max_prefix_len: Optional[int] = None,
        apply_chat_template_kwargs=None,
        success_reward_threshold: Optional[float] = None,
    ):
        self.tokenizer = tokenizer
        self.max_prefix_len = max_prefix_len
        self.apply_chat_template_kwargs = dict(apply_chat_template_kwargs or {})
        self.success_reward_threshold = success_reward_threshold

    @abstractmethod
    def build(self, inputs: TeacherInputs) -> TeacherBatch:
        """The teacher fields and metrics for one batch."""

    def trajectory_metrics(
        self, batch: TeacherBatch, inputs: TeacherInputs, supervised_per_row: list[float], weights: list[float]
    ) -> dict:
        """The teacher's own metrics that need the trajectory grouping (``inputs.traj_of_row``)
        and the final row weights; the trainer calls it after weighting the batch it built."""
        return {}


def make_teacher(cfg, tokenizer, max_prefix_len: int, apply_chat_template_kwargs=None) -> SDPOTeacher:
    """Instantiate the teacher ``cfg.teacher`` names by its ``_target_`` (a config file under
    ``trainer/config/sdpo_teacher/``), handing it what the trainer provides for every teacher."""
    teacher_cfg = cfg.teacher
    if not teacher_cfg or "_target_" not in teacher_cfg:
        raise ValueError(
            "self_distillation.teacher must carry a _target_ naming an SDPOTeacher class "
            f"(see trainer/config/sdpo_teacher/), got {teacher_cfg!r}"
        )
    try:
        teacher = hydra.utils.instantiate(
            teacher_cfg,
            tokenizer=tokenizer,
            max_prefix_len=max_prefix_len,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
            success_reward_threshold=cfg.success_reward_threshold,
            _recursive_=False,
            _convert_="all",
        )
    except InstantiationException as exc:
        # the constructor's own TypeError (an unknown key) or ValueError names the option
        if exc.__cause__ is None:
            raise
        raise exc.__cause__ from exc
    if not isinstance(teacher, SDPOTeacher):
        raise TypeError(f"self_distillation.teacher._target_ must build an SDPOTeacher, got {type(teacher).__name__}")
    return teacher
