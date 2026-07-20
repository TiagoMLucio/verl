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
"""SDPO teacher-context construction.

Per-sample selection of the privileged context (successful-sibling demo, environment
feedback) and assembly of the teacher's reprompt messages. Pure functions over an explicit
:class:`TeacherSampleContext`; the trainers own the batch plumbing (fetching rollout data,
tokenization, tensor assembly).
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

__all__ = [
    "TeacherSampleContext",
    "build_teacher_messages",
    "build_spliced_teacher_row",
    "collect_feedback",
    "collect_solutions_by_uid",
    "extract_prompt_text",
    "feedback_used",
    "remove_thinking_trace",
    "select_hinted_turns",
    "select_solution",
    "turn_token_mask",
]


@dataclass
class TeacherSampleContext:
    """Everything the teacher-message builder needs to know about one sample."""

    raw_prompt: list[dict]
    prompt_text: str
    solution: Optional[str] = None
    feedback: Optional[str] = None
    extra_fields: dict = field(default_factory=dict)


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


def collect_solutions_by_uid(
    uids: list[Any], reward_tensor: torch.Tensor, success_reward_threshold: float
) -> dict[Any, list[int]]:
    """Collect successful sample indices per UID based on sequence-level reward threshold."""
    seq_scores = reward_tensor.sum(dim=-1).detach().cpu().numpy()
    success_by_uid: dict[Any, list[int]] = defaultdict(list)
    for idx, uid in enumerate(uids):
        if seq_scores[idx] >= success_reward_threshold:
            success_by_uid[uid].append(idx)
    return success_by_uid


def remove_thinking_trace(text: str) -> str:
    """Remove <think>...</think> sections from a demonstration."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def select_solution(
    idx: int,
    success_by_uid: dict[Any, list[int]],
    uids: list[Any],
    response_texts: list[str],
    dont_reprompt_on_self_success: bool = False,
    remove_thinking_from_demonstration: bool = False,
) -> Optional[str]:
    """Select a successful demonstration for one sample from its UID group."""
    uid = uids[idx]
    solution_idxs = success_by_uid[uid]
    if dont_reprompt_on_self_success:
        solution_idxs = [j for j in solution_idxs if j != idx]
    if len(solution_idxs) == 0:
        return None
    solution_str = response_texts[solution_idxs[0]]
    if remove_thinking_from_demonstration:
        solution_str = remove_thinking_trace(solution_str)
    return solution_str


def extract_prompt_text(raw_prompt: list[dict]) -> str:
    """The single-turn user prompt text the reprompt template rebuilds around."""
    if len(raw_prompt) == 0:
        return ""
    content = raw_prompt[-1].get("content", "")
    if not isinstance(content, str):
        raise ValueError("SDPO currently only supports textual single-turn prompts.")
    return content


def select_hinted_turns(
    extra_fields: dict, response_len: int, max_hinted_turns: Optional[int] = None
) -> list[tuple[int, int, int, str]]:
    """Pair a sample's turn spans with its reflection diagnoses as (step, start, end, text).

    Spans are clamped to the (possibly truncated) response; with a cap, the first
    ``max_hinted_turns`` turns are kept (earliest, before the trajectory loses coherence).
    """
    diagnoses = {int(step): text for step, text in (extra_fields.get("turn_feedback") or [])}
    hinted = []
    for step, start, end in extra_fields.get("turn_spans") or []:
        step, start, end = int(step), int(start), min(int(end), response_len)
        if step in diagnoses and start < end:
            hinted.append((step, start, end, diagnoses[step]))
    if max_hinted_turns is not None and len(hinted) > max_hinted_turns:
        hinted = hinted[:max_hinted_turns]
    return hinted


def build_spliced_teacher_row(
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    hinted_turns: list[tuple[int, int, int, str]],
    hint_ids_list: list[torch.Tensor],
    max_prefix_len: int,
    header_ids: torch.Tensor,
) -> tuple[torch.Tensor, list[int], int]:
    """One teacher sequence per sample: each hint (a rendered user turn) inserted immediately
    before its turn's assistant header, truncated after the last hinted span. Inserting before
    the header keeps each hinted turn starting from the normal turn-initial state; later turns
    see earlier hints. When the header is not found before the span, the hint falls back to a
    bare splice at the span start; fallbacks are counted and returned.

    The prompt is left-truncated to ``max_prefix_len``; the body (everything after it) is the
    sequence's verbatim tail, so the worker's length-subtraction contract holds. Returns the
    sequence, its flat meta [body_len, then (body_start, start, end) per span] mapping each
    response span [start, end) to its position inside the body, and the fallback count.
    """
    prefix = prompt_ids if prompt_ids.shape[0] <= max_prefix_len else prompt_ids[-max_prefix_len:]
    chunks: list[torch.Tensor] = []
    meta: list[int] = []
    cursor = body_len = 0
    for (_, start, end, _), hint_ids in zip(hinted_turns, hint_ids_list, strict=True):
        assert cursor <= start, f"hinted turns must be ordered and disjoint: cursor {cursor} > start {start}"
        chunks.append(response_ids[cursor:start])
        body_len += start - cursor
        chunks.append(hint_ids.to(response_ids.dtype))
        body_len += hint_ids.shape[0]
        meta.extend([body_len, start, end])
        chunks.append(response_ids[start:end])
        body_len += end - start
        cursor = end
    return torch.cat([prefix, *chunks]), [body_len, *meta]


def turn_token_mask(response_len: int, hinted_turns: list[tuple[int, int, int, str]]) -> torch.Tensor:
    """Per-token distillation mask: 1 on hinted turn spans, 0 elsewhere."""
    mask = torch.zeros(response_len, dtype=torch.float32)
    for _, start, end, _ in hinted_turns:
        mask[start:end] = 1.0
    return mask


def feedback_used(ctx: TeacherSampleContext, cfg) -> bool:
    """Whether this sample's feedback enters the teacher context (mirrors the mask)."""
    feedback_only_without_solution = cfg.get("environment_feedback_only_without_solution", False)
    return ctx.feedback is not None and (not feedback_only_without_solution or ctx.solution is None)


def build_teacher_messages(ctx: TeacherSampleContext, cfg) -> list[dict]:
    """Assemble the teacher's reprompt messages for one sample."""
    has_solution = ctx.solution is not None
    use_feedback = feedback_used(ctx, cfg)

    solution_section = ""
    if has_solution:
        solution_section = cfg.solution_template.format(successful_previous_attempt=ctx.solution)

    feedback_section = ""
    if use_feedback:
        feedback_section = cfg.feedback_template.format(feedback_raw=ctx.feedback)

    # Per-segment teacher context: a condensation segment (segment_index > 0) was generated
    # from a *condensed history*, not the original task. Build its teacher from that history
    # + an appended augmentation turn, so teacher and student share the same context
    # (differing only by feedback/solution). Segment 0 (and the no-condensation case) keeps
    # the original single-turn reconstruction below.
    segment_prompt = ctx.extra_fields.get("segment_prompt")
    if segment_prompt and ctx.extra_fields.get("segment_index", 0):
        if not (use_feedback or has_solution):
            return list(segment_prompt)
        aug_text = cfg.reprompt_template.format(prompt="", solution=solution_section, feedback=feedback_section)
        return list(segment_prompt) + [{"role": "user", "content": aug_text}]

    system_messages = ctx.raw_prompt[:-1]
    if use_feedback or has_solution:
        reprompt_text = cfg.reprompt_template.format(
            prompt=ctx.prompt_text, solution=solution_section, feedback=feedback_section
        )
    else:
        reprompt_text = ctx.prompt_text

    return system_messages + [{"role": "user", "content": reprompt_text}]
