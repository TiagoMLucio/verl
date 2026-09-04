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
"""The turn-hint teacher: hints only, no sibling solution, no response decode."""

import torch

from verl.trainer.ppo.sdpo.batch import TeacherBatch, TeacherInputs
from verl.trainer.ppo.sdpo.hints import assistant_header_ids, hint_token_ids, select_hinted_turns
from verl.trainer.ppo.sdpo.splice import build_spliced_teacher_row, turn_token_mask
from verl.trainer.ppo.sdpo.teacher_meta import DEGENERATE_META


class TurnHintTeacher:
    """Supervision is hints-only: a sample carrying reflection hints ships one spliced teacher
    sequence (each hint inserted before its turn, ``teacher_seq_meta`` mapping the scored spans
    back to the response grid) and a per-token distillation mask over those spans. Un-hinted
    samples are not trained at all: a degenerate 2-token teacher row (1-token body, ``DEGENERATE_META``)
    with a zero mask, scored only so that dp-group collectives stay in lockstep.

    ``max_prefix_len`` caps the spliced prefix, which is the student's real prompt (segment
    rows reach ~24k): the student's own prompt budget, not the reprompt one.
    """

    needs_prompts = True

    def __init__(self, tokenizer, cfg, max_prefix_len: int):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.max_prefix_len = max_prefix_len
        self.template_kwargs = dict(cfg.chat_template_kwargs)
        self.header_ids = torch.tensor(
            assistant_header_ids(tokenizer, template_kwargs=self.template_kwargs), dtype=torch.int64
        )
        # call-placed splices close the assistant turn and reopen it after the hint; the
        # call span starts at the template's tool-call opening token
        self.close_ids = torch.tensor(
            tokenizer.encode(tokenizer.eos_token + "\n", add_special_tokens=False), dtype=torch.int64
        )
        self.call_open_ids = torch.tensor(tokenizer.encode("<tool_call>", add_special_tokens=False), dtype=torch.int64)

    def build(self, inputs: TeacherInputs) -> TeacherBatch:
        from verl.utils.debug_breakpoints import should_break

        cfg = self.cfg
        hinted_per_row = [
            select_hinted_turns(extra_fields, response.shape[0], cfg.max_hinted_turns)
            for extra_fields, response in zip(inputs.extra_fields, inputs.responses, strict=True)
        ]
        teacher_seqs, seq_meta, mask_rows, loss_mask_rows = [], [], [], []
        hint_fallbacks = 0
        for prompt_ids, response_ids, response_mask, hinted in zip(
            inputs.prompts, inputs.responses, inputs.response_mask, hinted_per_row, strict=True
        ):
            if hinted:
                if should_break("teacher_build_row"): breakpoint()
                hint_ids = [hint_token_ids(self.tokenizer, hint, cfg, self.template_kwargs) for hint in hinted]
                seq, meta, fallbacks, spans = build_spliced_teacher_row(
                    prompt_ids,
                    response_ids,
                    hinted,
                    hint_ids,
                    self.max_prefix_len,
                    self.header_ids,
                    close_ids=self.close_ids,
                    call_open_ids=self.call_open_ids,
                )
                hint_fallbacks += fallbacks
                mask_row = turn_token_mask(response_ids.shape[0], spans)
            else:
                seq = torch.cat([prompt_ids[-1:], response_ids[:1]])
                meta = DEGENERATE_META
                mask_row = torch.zeros(response_ids.shape[0], dtype=torch.float32)
            teacher_seqs.append(seq)
            seq_meta.append(torch.tensor(meta, dtype=torch.int64))
            mask_rows.append(mask_row)
            loss_mask_rows.append(response_mask * mask_row.to(response_mask.dtype))

        fields = {
            "teacher_input_ids": torch.nested.nested_tensor(teacher_seqs, layout=torch.jagged),
            "teacher_seq_meta": torch.nested.nested_tensor(seq_meta, layout=torch.jagged),
            "self_distillation_mask": torch.nested.nested_tensor(mask_rows, layout=torch.jagged),
            "loss_mask": torch.nested.nested_tensor(loss_mask_rows, layout=torch.jagged),
        }
        num_hinted = sum(1 for hinted in hinted_per_row if hinted)
        metrics = {
            "self_distillation/hinted_sample_fraction": num_hinted / len(inputs),
            "self_distillation/hinted_turns_per_sample": (
                sum(len(hinted) for hinted in hinted_per_row) / num_hinted if num_hinted else 0.0
            ),
            "self_distillation/hint_injection_fallbacks": hint_fallbacks,
            "self_distillation/call_loss_weight": float(cfg.call_loss_weight),
        }
        return TeacherBatch(fields=fields, metrics=metrics, hinted_per_row=hinted_per_row)
