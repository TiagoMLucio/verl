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
"""The paper's reprompt teacher context: a sibling solution and the environment's feedback
folded into a fresh single-turn prompt. Pure functions over :class:`RepromptContext`; the
teachers own the batch plumbing."""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

__all__ = [
    "RepromptContext",
    "build_reprompt_messages",
    "collect_feedback",
    "extract_prompt_text",
    "prompt_feedback_used",
    "remove_thinking_trace",
    "segment_prompt_of",
    "select_solution_row",
    "success_rows_by_uid",
    "tokenize_reprompt_batch",
]


def extract_prompt_text(raw_prompt: list[dict]) -> str:
    """The single-turn user prompt text the reprompt template rebuilds around."""
    if len(raw_prompt) == 0:
        return ""
    content = raw_prompt[-1].get("content", "")
    if not isinstance(content, str):
        raise ValueError("SDPO currently only supports textual single-turn prompts.")
    return content


@dataclass
class RepromptContext:
    """One sample's side of the reprompt: its prompt, the feedback its reward produced and, for
    a condensation segment past the first, the condensed history it was generated from."""

    raw_prompt: list[dict]
    feedback: Optional[str] = None
    segment_prompt: Optional[list[dict]] = None

    @property
    def prompt_text(self) -> str:
        return extract_prompt_text(self.raw_prompt)


def segment_prompt_of(extra_fields: dict) -> Optional[list[dict]]:
    """The condensed history a segment past the first was generated from; None for segment 0."""
    return extra_fields.get("segment_prompt") if extra_fields.get("segment_index", 0) else None


def collect_feedback(
    include_environment_feedback: bool,
    reward_extra_infos_dict: Optional[dict[str, Any]],
    batch_size: int,
) -> list[Any]:
    """Collect non-empty textual environment feedback from reward extras."""
    feedback_list: list[Any] = [None] * batch_size
    if include_environment_feedback and reward_extra_infos_dict is not None:
        raw_feedback = reward_extra_infos_dict.get("feedback", [])
        if isinstance(raw_feedback, np.ndarray):
            raw_feedback = raw_feedback.tolist()
        for i in range(min(len(raw_feedback), batch_size)):
            if isinstance(raw_feedback[i], str) and raw_feedback[i].strip():
                feedback_list[i] = raw_feedback[i]
    return feedback_list


def success_rows_by_uid(
    uids: list[Any], seq_scores: list[float], success_reward_threshold: float
) -> dict[Any, list[int]]:
    """Row indices of the successful samples of each UID group, by sequence-level reward."""
    success_by_uid: dict[Any, list[int]] = defaultdict(list)
    for idx, uid in enumerate(uids):
        if seq_scores[idx] >= success_reward_threshold:
            success_by_uid[uid].append(idx)
    return success_by_uid


def remove_thinking_trace(text: str) -> str:
    """Remove <think>...</think> sections from a demonstration."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def select_solution_row(
    idx: int,
    success_by_uid: dict[Any, list[int]],
    uids: list[Any],
    dont_reprompt_on_self_success: bool = False,
) -> Optional[int]:
    """Row index of the successful demonstration for one sample from its UID group, or None."""
    solution_idxs = success_by_uid[uids[idx]]
    if dont_reprompt_on_self_success:
        solution_idxs = [j for j in solution_idxs if j != idx]
    return solution_idxs[0] if solution_idxs else None


def prompt_feedback_used(feedback: Optional[str], has_solution: bool, cfg) -> bool:
    """Whether this sample's feedback enters the teacher prompt (mirrors the mask)."""
    return feedback is not None and (not cfg.environment_feedback_only_without_solution or not has_solution)


def build_reprompt_messages(ctx: RepromptContext, solution: Optional[str], cfg) -> list[dict]:
    """Assemble the teacher's reprompt messages for one sample."""
    has_solution = solution is not None
    use_feedback = prompt_feedback_used(ctx.feedback, has_solution, cfg)

    solution_section = ""
    if has_solution:
        solution_section = cfg.solution_template.format(successful_previous_attempt=solution)

    feedback_section = ""
    if use_feedback:
        feedback_section = cfg.feedback_template.format(feedback_raw=ctx.feedback)

    # Per-segment teacher context: a condensation segment (segment_index > 0) was generated
    # from a *condensed history*, not the original task. Build its teacher from that history
    # + an appended augmentation turn, so teacher and student share the same context
    # (differing only by feedback/solution). Segment 0 (and the no-condensation case) keeps
    # the original single-turn reconstruction below.
    if ctx.segment_prompt:
        if not (use_feedback or has_solution):
            return list(ctx.segment_prompt)
        aug_text = cfg.reprompt_template.format(prompt="", solution=solution_section, feedback=feedback_section)
        return list(ctx.segment_prompt) + [{"role": "user", "content": aug_text}]

    system_messages = ctx.raw_prompt[:-1]
    if use_feedback or has_solution:
        reprompt_text = cfg.reprompt_template.format(
            prompt=ctx.prompt_text, solution=solution_section, feedback=feedback_section
        )
    else:
        reprompt_text = ctx.prompt_text

    return system_messages + [{"role": "user", "content": reprompt_text}]


def tokenize_reprompt_batch(
    tokenizer, messages: list[list[dict]], cfg, apply_chat_template_kwargs=None
) -> list[torch.Tensor]:
    """Tokenize the reprompts as one left-padded batch capped at ``cfg.max_reprompt_len``
    (truncated on ``cfg.reprompt_truncation``) and return each row without its padding."""
    if not messages:
        return []
    apply_kwargs = dict(
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
        max_length=cfg.max_reprompt_len,
        padding=True,
        truncation=True,
        **dict(apply_chat_template_kwargs or {}),
    )
    sides = tokenizer.padding_side, tokenizer.truncation_side
    tokenizer.padding_side = "left"
    if cfg.reprompt_truncation in {"left", "right"}:
        tokenizer.truncation_side = cfg.reprompt_truncation
    try:
        try:
            teacher_prompt = tokenizer.apply_chat_template(messages, continue_final_message=False, **apply_kwargs)
        except TypeError:
            teacher_prompt = tokenizer.apply_chat_template(messages, **apply_kwargs)
    finally:
        tokenizer.padding_side, tokenizer.truncation_side = sides

    if isinstance(teacher_prompt, torch.Tensor):
        input_ids = teacher_prompt
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        attention_mask = (input_ids != pad_token_id).to(dtype=torch.long)
    else:
        input_ids = teacher_prompt["input_ids"]
        attention_mask = teacher_prompt.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return [ids[mask.bool()] for ids, mask in zip(input_ids, attention_mask, strict=True)]
