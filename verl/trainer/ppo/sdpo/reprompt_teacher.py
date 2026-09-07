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
"""The paper's reprompt teacher: a fresh prompt carrying the sibling solution and the
environment feedback, followed by the student's own response."""

from typing import Optional

import torch

from verl.trainer.ppo.sdpo.batch import TeacherBatch, TeacherInputs
from verl.trainer.ppo.sdpo.reprompt import (
    RepromptContext,
    build_reprompt_messages,
    prompt_feedback_used,
    remove_thinking_trace,
    segment_prompt_of,
    select_solution_row,
    success_rows_by_uid,
    tokenize_reprompt_batch,
)
from verl.trainer.ppo.sdpo.teacher import SDPOTeacher


class RepromptTeacher(SDPOTeacher):
    """Every row is supervised whose group holds a successful sibling or whose reward produced
    feedback: its teacher sequence is the reprompt (left padding stripped) followed by the
    response, and its mask is the whole response. Only the responses chosen as a solution are
    decoded.

    The keyword options are ``trainer/config/sdpo_teacher/reprompt.yaml``; their defaults here
    are that file's.
    """

    def __init__(
        self,
        tokenizer,
        *,
        max_prefix_len: Optional[int] = None,
        apply_chat_template_kwargs=None,
        success_reward_threshold: float,
        max_reprompt_len: int = 10240,
        reprompt_truncation: str = "right",
        dont_reprompt_on_self_success: bool = True,
        remove_thinking_from_demonstration: bool = True,
        reprompt_template: str = "{prompt}{solution}{feedback}\n\nCorrectly solve the original question.\n",
        solution_template: str = "\nCorrect solution:\n\n{successful_previous_attempt}\n\n",
        feedback_template: str = (
            "\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}\n\n"
        ),
        environment_feedback_only_without_solution: bool = True,
    ):
        super().__init__(
            tokenizer,
            max_prefix_len=max_prefix_len,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
            success_reward_threshold=success_reward_threshold,
        )
        if reprompt_truncation not in ("right", "left"):
            raise ValueError(f"reprompt_truncation must be right|left, got {reprompt_truncation!r}")
        self.max_reprompt_len = max_reprompt_len
        self.reprompt_truncation = reprompt_truncation
        self.dont_reprompt_on_self_success = dont_reprompt_on_self_success
        self.remove_thinking_from_demonstration = remove_thinking_from_demonstration
        self.reprompt_template = reprompt_template
        self.solution_template = solution_template
        self.feedback_template = feedback_template
        self.environment_feedback_only_without_solution = environment_feedback_only_without_solution

    def solution_text(self, response_ids: torch.Tensor) -> str:
        text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return remove_thinking_trace(text) if self.remove_thinking_from_demonstration else text

    def build(self, inputs: TeacherInputs) -> TeacherBatch:
        success_by_uid = success_rows_by_uid(inputs.uids, inputs.seq_scores, self.success_reward_threshold)
        solution_row = [
            select_solution_row(i, success_by_uid, inputs.uids, self.dont_reprompt_on_self_success)
            for i in range(len(inputs))
        ]
        solution_text = {row: self.solution_text(inputs.responses[row]) for row in set(solution_row) - {None}}
        contexts = [
            RepromptContext(raw_prompt=raw_prompt, feedback=feedback, segment_prompt=segment_prompt_of(extra_fields))
            for raw_prompt, feedback, extra_fields in zip(
                inputs.raw_prompts, inputs.feedback, inputs.extra_fields, strict=True
            )
        ]
        messages = [
            build_reprompt_messages(ctx, None if row is None else solution_text[row], self)
            for ctx, row in zip(contexts, solution_row, strict=True)
        ]
        prompts = tokenize_reprompt_batch(self.tokenizer, messages, self, self.apply_chat_template_kwargs)
        reprompt_mask = [
            row is not None
            or prompt_feedback_used(ctx.feedback, row is not None, self.environment_feedback_only_without_solution)
            for ctx, row in zip(contexts, solution_row, strict=True)
        ]
        fields = {
            "teacher_input_ids": torch.nested.nested_tensor(
                [torch.cat([prompt, response]) for prompt, response in zip(prompts, inputs.responses, strict=True)],
                layout=torch.jagged,
            ),
            "self_distillation_mask": torch.tensor(reprompt_mask, dtype=torch.float32),
            "loss_mask": torch.nested.nested_tensor(
                [mask * int(used) for mask, used in zip(inputs.response_mask, reprompt_mask, strict=True)],
                layout=torch.jagged,
            ),
        }
        return TeacherBatch(fields=fields)

    def supervision_source_metrics(self, inputs: TeacherInputs, traj_of_row: list) -> dict:
        from verl.trainer.ppo.sdpo_health_metrics import supervision_source_metrics

        return supervision_source_metrics(
            inputs.uids, inputs.seq_scores, inputs.feedback, inputs.extra_fields, traj_of_row,
            self.success_reward_threshold,
            dont_reprompt_on_self_success=self.dont_reprompt_on_self_success,
            environment_feedback_only_without_solution=self.environment_feedback_only_without_solution,
        )
