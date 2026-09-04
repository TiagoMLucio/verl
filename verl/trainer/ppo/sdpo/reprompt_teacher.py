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


class RepromptTeacher:
    """Every row is supervised whose group holds a successful sibling or whose reward produced
    feedback: its teacher sequence is the reprompt (left padding stripped) followed by the
    response, and its mask is the whole response. Only the responses chosen as a solution are
    decoded."""

    needs_prompts = False

    def __init__(self, tokenizer, cfg, apply_chat_template_kwargs=None):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.apply_chat_template_kwargs = dict(apply_chat_template_kwargs or {})

    def solution_text(self, response_ids: torch.Tensor) -> str:
        text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return remove_thinking_trace(text) if self.cfg.remove_thinking_from_demonstration else text

    def build(self, inputs: TeacherInputs) -> TeacherBatch:
        cfg = self.cfg
        success_by_uid = success_rows_by_uid(inputs.uids, inputs.seq_scores, cfg.success_reward_threshold)
        solution_row = [
            select_solution_row(i, success_by_uid, inputs.uids, cfg.dont_reprompt_on_self_success)
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
            build_reprompt_messages(ctx, None if row is None else solution_text[row], cfg)
            for ctx, row in zip(contexts, solution_row, strict=True)
        ]
        prompts = tokenize_reprompt_batch(self.tokenizer, messages, cfg, self.apply_chat_template_kwargs)
        reprompt_mask = [
            row is not None or prompt_feedback_used(ctx.feedback, row is not None, cfg)
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
