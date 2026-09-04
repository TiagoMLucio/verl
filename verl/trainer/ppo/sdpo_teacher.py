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

The turn-hint path pairs reflection hints with their turns (:class:`HintedTurn`) and builds
the spliced teacher row: one sub-row per hint, its meta packed by
:mod:`verl.trainer.ppo.sdpo.teacher_meta`.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional

import numpy as np
import torch

from verl.trainer.ppo.sdpo.teacher_meta import SubRow, pack

__all__ = [
    "HintedTurn",
    "TeacherSampleContext",
    "build_teacher_messages",
    "build_spliced_teacher_row",
    "collect_feedback",
    "collect_solutions_by_uid",
    "extract_prompt_text",
    "feedback_used",
    "hint_token_ids",
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


class HintedTurn(NamedTuple):
    """One reflection hint paired with the turn it lands on: ``[start, end)`` on the response
    grid, spliced before the whole turn (``placement == "turn"``, the default) or between the
    turn's reasoning and its tool call (``"call"``)."""

    step: int
    start: int
    end: int
    text: str
    placement: str = "turn"

    @property
    def is_call(self) -> bool:
        return self.placement == "call"


def select_hinted_turns(
    extra_fields: dict, response_len: int, max_hinted_turns: Optional[int] = None
) -> list[HintedTurn]:
    """Pair a sample's turn spans with its reflection diagnoses. The rollout ships the
    placement as an optional third element of the ``turn_feedback`` entry; a fourth element
    (the ``target`` field older rollout dumps still carry) is ignored.

    Spans are clamped to the (possibly truncated) response; with a cap, the first
    ``max_hinted_turns`` turns are kept (earliest, before the trajectory loses coherence).
    """
    diagnoses = {int(entry[0]): (entry[1], entry[2] if len(entry) > 2 else "turn")
                 for entry in (extra_fields.get("turn_feedback") or [])}
    hinted = []
    for step, start, end in extra_fields.get("turn_spans") or []:
        step, start, end = int(step), int(start), min(int(end), response_len)
        if step in diagnoses and start < end:
            text, placement = diagnoses[step]
            hinted.append(HintedTurn(step, start, end, text, placement))
    if max_hinted_turns is not None and len(hinted) > max_hinted_turns:
        hinted = hinted[:max_hinted_turns]
    return hinted


# Render-suffix over this probe yields the exact mid-conversation fragment (auto system blocks cancel in the prefix).
_TEMPLATE_PROBE = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
# User-turn fragments use a user-only probe: templates that re-render a no-longer-final
# assistant turn (Qwen3.5 drops its empty think block) break the two-turn probe's prefix
# property, while a trailing user turn never changes how the probe itself renders.
_TEMPLATE_PROBE_USER = [{"role": "user", "content": "x"}]


def _template_suffix(
    tokenizer, messages=(), add_generation_prompt=False, probe=_TEMPLATE_PROBE, template_kwargs=None
) -> str:
    kwargs = dict(template_kwargs or {})
    base = tokenizer.apply_chat_template(list(probe), tokenize=False, add_generation_prompt=False, **kwargs)
    full = tokenizer.apply_chat_template(
        list(probe) + list(messages), tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs
    )
    assert full.startswith(base), "chat template does not render conversations as extendable prefixes"
    return full[len(base) :]


def assistant_header_ids(tokenizer, template_kwargs=None) -> list[int]:
    """Token ids of the template's assistant generation header (e.g. ``<|im_start|>assistant\\n``).

    ``template_kwargs`` must match the kwargs the rollout passed to ``apply_chat_template``
    (e.g. ``{"enable_thinking": False}``), or the header will not match the rollout tokens.
    """
    return tokenizer.encode(
        _template_suffix(tokenizer, add_generation_prompt=True, template_kwargs=template_kwargs),
        add_special_tokens=False,
    )


def hint_user_turn_ids(tokenizer, hint_text: str, template_kwargs=None) -> list[int]:
    """Token ids of ``hint_text`` rendered as a full user turn of the tokenizer's chat template."""
    suffix = _template_suffix(
        tokenizer,
        messages=[{"role": "user", "content": hint_text}],
        probe=_TEMPLATE_PROBE_USER,
        template_kwargs=template_kwargs,
    )
    return tokenizer.encode(suffix, add_special_tokens=False)


def hint_token_ids(tokenizer, hint: HintedTurn, cfg, template_kwargs=None) -> torch.Tensor:
    """Token ids of ``hint`` wrapped in the template its placement calls for
    (``cfg.call_feedback_template`` or ``cfg.turn_feedback_template``), rendered as a user turn."""
    template = cfg.call_feedback_template if hint.is_call else cfg.turn_feedback_template
    return torch.tensor(
        hint_user_turn_ids(tokenizer, template.format(diagnosis=hint.text), template_kwargs=template_kwargs),
        dtype=torch.int64,
    )


def _find_subseq(haystack: torch.Tensor, needle: torch.Tensor, start: int, end: int) -> Optional[int]:
    """Index of the first occurrence of ``needle`` inside ``haystack[start:end]``, or None."""
    n = needle.shape[0]
    if n == 0 or end - start < n:
        return None
    window = haystack[start:end]
    hits = (window.unfold(0, n, 1) == needle).all(dim=1).nonzero()
    return start + hits[0].item() if len(hits) else None


def _sub_row(
    prefix: torch.Tensor, body: list[torch.Tensor], scored: int, span: tuple[int, int]
) -> tuple[torch.Tensor, SubRow]:
    """Concatenate one sub-row from its prefix and ordered body pieces; ``body[scored]`` is
    the span the teacher scores."""
    body_len = sum(part.shape[0] for part in body)
    body_start = sum(part.shape[0] for part in body[:scored])
    return torch.cat([prefix, *body]), SubRow(prefix.shape[0] + body_len, body_len, body_start, *span)


def build_spliced_teacher_row(
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    hinted_turns: list[HintedTurn],
    hint_ids_list: list[torch.Tensor],
    max_prefix_len: int,
    header_ids: torch.Tensor,
    close_ids: Optional[torch.Tensor] = None,
    call_open_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, list[int], int, list[tuple[int, int]]]:
    """One teacher sub-row per hinted turn, concatenated into a single row.

    Each sub-row is the trajectory up to its own turn with only its own hint spliced in,
    and truncated after the scored span. Turn-placed hints go immediately before the
    turn's assistant header and score the whole turn. Call-placed hints go between
    the turn's reasoning and its tool call: the assistant turn is closed (``close_ids``),
    the hint's user turn inserted, the assistant header reopened, and only the call span
    (from ``call_open_ids``, e.g. the ``<tool_call>`` token, to the turn's end) is scored.
    A call hint whose turn has no call opening falls back to the turn splice.

    Carrying every hint in one sequence would make the teacher score a later turn from a
    state it could not reach: it would see its own earlier advice followed by the student
    ignoring it.

    The prompt is left-truncated to ``max_prefix_len``. Returns the concatenation, the packed
    meta (see :mod:`verl.trainer.ppo.sdpo.teacher_meta`), the number of fallback placements
    and the (start, end) spans for the distillation mask (== the meta spans).
    """
    base_prefix = prompt_ids if prompt_ids.shape[0] <= max_prefix_len else prompt_ids[-max_prefix_len:]
    header = header_ids.shape[0]
    pieces: list[torch.Tensor] = []
    sub_rows: list[SubRow] = []
    fallbacks = 0

    for hint, hint_ids in zip(hinted_turns, hint_ids_list, strict=True):
        start, end = hint.start, hint.end
        hint_ids = hint_ids.to(response_ids.dtype)
        prefix = base_prefix
        call_at = None
        if hint.is_call and close_ids is not None and call_open_ids is not None:
            call_at = _find_subseq(response_ids, call_open_ids.to(response_ids.dtype), start, end)

        if call_at is not None:
            body = [
                response_ids[:call_at],  # history plus this turn's header and reasoning
                close_ids.to(response_ids.dtype),
                hint_ids,
                header_ids.to(response_ids.dtype),
                response_ids[call_at:end],  # the call, the only span the teacher scores
            ]
            row, sub_row = _sub_row(prefix, body, scored=4, span=(call_at, end))
        else:
            # one count per hint, whether the call opening was missing, the header was, or both
            degraded = hint.is_call
            if start >= header and torch.equal(response_ids[start - header : start], header_ids):
                insert_at = start - header
            elif start == 0 and prefix.shape[0] >= header and torch.equal(prefix[-header:], header_ids):
                # first turn: its assistant header is the prompt tail, so the hint joins the prefix
                prefix = torch.cat([prefix[:-header], hint_ids, prefix[-header:]])
                insert_at = None
            else:
                insert_at = start
                degraded = True
            fallbacks += int(degraded)

            if insert_at is None:
                body = [response_ids[:start], response_ids[start:end]]
                row, sub_row = _sub_row(prefix, body, scored=1, span=(start, end))
            else:
                body = [
                    response_ids[:insert_at],  # untouched history, no other hints
                    hint_ids,
                    response_ids[insert_at:start],  # the turn's assistant header
                    response_ids[start:end],  # the span the teacher scores
                ]
                row, sub_row = _sub_row(prefix, body, scored=3, span=(start, end))

        pieces.append(row)
        sub_rows.append(sub_row)

    return torch.cat(pieces), pack(sub_rows), fallbacks, [(sub_row.start, sub_row.end) for sub_row in sub_rows]


def trace_weights(
    supervised_per_row: list[float],
    traj_of_row: list,
    call_row: list[bool],
    call_weight: float = 1.0,
) -> list[float]:
    """Per-row weight for the seq-mean loss: a trajectory counts once in total, its segments
    split that weight by how much supervision each carries.

    ``call_weight`` (lambda) rescales rows supervised by a mid-turn call hint relative to
    rows supervised by turn-level (pipeline) hints. It belongs here rather than on the token
    mask because the per-row loss is a token-mean, in which a uniform within-row scale
    cancels; a trajectory carries one kind of hint or the other, so the mix is across rows.
    Weights are renormalised to the supervised-row count, so lambda re-allocates influence
    between the two channels without changing the update's overall scale.
    """
    traj_supervised: dict = defaultdict(float)
    for traj, n_supervised in zip(traj_of_row, supervised_per_row, strict=True):
        traj_supervised[traj] += n_supervised
    weights = [
        (n / traj_supervised[traj] if traj_supervised[traj] > 0 else 0.0)
        * (call_weight if is_call else 1.0)
        for traj, n, is_call in zip(traj_of_row, supervised_per_row, call_row, strict=True)
    ]
    n_supervised_rows = sum(1 for n in supervised_per_row if n > 0)
    total = sum(weights)
    scale = (n_supervised_rows / total) if total > 0 else 1.0
    return [w * scale for w in weights]


def turn_token_mask(response_len: int, spans: list[tuple[int, int]]) -> torch.Tensor:
    """Per-token distillation mask: 1 on the scored spans, 0 elsewhere."""
    mask = torch.zeros(response_len, dtype=torch.float32)
    for start, end in spans:
        mask[start:end] = 1.0
    return mask


def feedback_used(ctx: TeacherSampleContext, cfg) -> bool:
    """Whether this sample's feedback enters the teacher context (mirrors the mask)."""
    return ctx.feedback is not None and (not cfg.environment_feedback_only_without_solution or ctx.solution is None)


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
