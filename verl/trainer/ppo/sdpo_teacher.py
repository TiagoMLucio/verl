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
) -> list[tuple[int, int, int, str, str, "Optional[str]"]]:
    """Pair a sample's turn spans with its reflection diagnoses as
    (step, start, end, text, at, target), where ``at`` is ``turn`` (hint before the whole turn,
    the default) or ``call`` (hint between the turn's reasoning and its tool call; the
    rollout ships it as a third element of the ``turn_feedback`` entry).

    Spans are clamped to the (possibly truncated) response; with a cap, the first
    ``max_hinted_turns`` turns are kept (earliest, before the trajectory loses coherence).
    """
    diagnoses = {int(entry[0]): (entry[1], entry[2] if len(entry) > 2 else "turn",
                                 entry[3] if len(entry) > 3 else None)
                 for entry in (extra_fields.get("turn_feedback") or [])}
    hinted = []
    for step, start, end in extra_fields.get("turn_spans") or []:
        step, start, end = int(step), int(start), min(int(end), response_len)
        if step in diagnoses and start < end:
            text, at, target = diagnoses[step]
            hinted.append((step, start, end, text, at, target))
    if max_hinted_turns is not None and len(hinted) > max_hinted_turns:
        hinted = hinted[:max_hinted_turns]
    return hinted


# Render-suffix over this probe yields the exact mid-conversation fragment (auto system blocks cancel in the prefix).
_TEMPLATE_PROBE = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]


def _template_suffix(tokenizer, messages=(), add_generation_prompt=False) -> str:
    base = tokenizer.apply_chat_template(_TEMPLATE_PROBE, tokenize=False, add_generation_prompt=False)
    full = tokenizer.apply_chat_template(
        _TEMPLATE_PROBE + list(messages), tokenize=False, add_generation_prompt=add_generation_prompt
    )
    assert full.startswith(base), "chat template does not render conversations as extendable prefixes"
    return full[len(base) :]


def assistant_header_ids(tokenizer) -> list[int]:
    """Token ids of the template's assistant generation header (e.g. ``<|im_start|>assistant\\n``)."""
    return tokenizer.encode(_template_suffix(tokenizer, add_generation_prompt=True), add_special_tokens=False)


def hint_user_turn_ids(tokenizer, hint_text: str) -> list[int]:
    """Token ids of ``hint_text`` rendered as a full user turn of the tokenizer's chat template."""
    suffix = _template_suffix(tokenizer, messages=[{"role": "user", "content": hint_text}])
    return tokenizer.encode(suffix, add_special_tokens=False)


def _find_subseq(haystack: torch.Tensor, needle: torch.Tensor, lo: int, hi: int) -> Optional[int]:
    """Index of the first occurrence of ``needle`` inside ``haystack[lo:hi]``, or None."""
    n = needle.shape[0]
    if n == 0 or hi - lo < n:
        return None
    window = haystack[lo:hi]
    hits = (window.unfold(0, n, 1) == needle).all(dim=1).nonzero()
    return lo + hits[0].item() if len(hits) else None


def build_spliced_teacher_row(
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    hinted_turns: list[tuple[int, int, int, str, str]],
    hint_ids_list: list[torch.Tensor],
    max_prefix_len: int,
    header_ids: torch.Tensor,
    close_ids: Optional[torch.Tensor] = None,
    call_open_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, list[int], int, list[tuple[int, int]], list[bool]]:
    """One teacher sub-row per hinted turn, concatenated into a single row.

    Each sub-row is the trajectory up to its own turn with only its own hint spliced in,
    and truncated after the scored span. ``at == "turn"`` hints go immediately before the
    turn's assistant header and score the whole turn. ``at == "call"`` hints go between
    the turn's reasoning and its tool call: the assistant turn is closed (``close_ids``),
    the hint's user turn inserted, the assistant header reopened, and only the call span
    (from ``call_open_ids``, e.g. the ``<tool_call>`` token, to the turn's end) is scored.
    A call hint whose turn has no call opening falls back to the turn splice.

    Carrying every hint in one sequence would make the teacher score a later turn from a
    state it could not reach: it would see its own earlier advice followed by the student
    ignoring it.

    The prompt is left-truncated to ``max_prefix_len``. Returns the concatenation, a flat
    meta ``[n_sub, (total_len, body_len, body_start, start, end) per sub-row]``, the
    number of fallback placements, the effective (start, end) spans for the
    distillation mask (== the meta spans), and per-hint whether the call splice was
    actually placed (False for turn hints and for call hints that fell back).
    """
    base_prefix = prompt_ids if prompt_ids.shape[0] <= max_prefix_len else prompt_ids[-max_prefix_len:]
    header = header_ids.shape[0]
    pieces: list[torch.Tensor] = []
    meta: list[int] = []
    spans: list[tuple[int, int]] = []
    call_placed: list[bool] = []
    fallbacks = 0

    for (_, start, end, _, at, *_), hint_ids in zip(hinted_turns, hint_ids_list, strict=True):
        hint_ids = hint_ids.to(response_ids.dtype)
        prefix = base_prefix
        call_at = None
        if at == "call" and close_ids is not None and call_open_ids is not None:
            call_at = _find_subseq(response_ids, call_open_ids.to(response_ids.dtype), start, end)

        if call_at is not None:
            body = [
                response_ids[:call_at],  # history plus this turn's header and reasoning
                close_ids.to(response_ids.dtype),
                hint_ids,
                header_ids.to(response_ids.dtype),
                response_ids[call_at:end],  # the call, the only span the teacher scores
            ]
            body_start = call_at + close_ids.shape[0] + hint_ids.shape[0] + header
            span = (call_at, end)
        else:
            if at == "call":
                fallbacks += 1
            if start >= header and torch.equal(response_ids[start - header : start], header_ids):
                insert_at = start - header
            elif start == 0 and prefix.shape[0] >= header and torch.equal(prefix[-header:], header_ids):
                # first turn: its assistant header is the prompt tail, so the hint joins the prefix
                prefix = torch.cat([prefix[:-header], hint_ids, prefix[-header:]])
                insert_at = None
            else:
                insert_at = start
                fallbacks += 1

            if insert_at is None:
                body = [response_ids[:start], response_ids[start:end]]
                body_start = start
            else:
                body = [
                    response_ids[:insert_at],  # untouched history, no other hints
                    hint_ids,
                    response_ids[insert_at:start],  # the turn's assistant header
                    response_ids[start:end],  # the span the teacher scores
                ]
                body_start = start + hint_ids.shape[0]
            span = (start, end)

        call_placed.append(call_at is not None)
        body_len = sum(part.shape[0] for part in body)
        pieces.append(torch.cat([prefix, *body]))
        meta.extend([prefix.shape[0] + body_len, body_len, body_start, *span])
        spans.append(span)

    return torch.cat(pieces), [len(hinted_turns), *meta], fallbacks, spans, call_placed


def divergence_spans(student_ids, target_ids, mode: str) -> list[tuple[int, int]]:
    """Positions (relative to the student call) where the corrected call changes tokens.

    Token-level diff via matching blocks, so independent errors at different offsets are
    found even when lengths shift. ``replace``/``delete`` blocks mask the student's wrong
    tokens; an ``insert`` (student missing tokens) masks the single boundary position that
    should have produced them. ``first`` keeps the first differing block, ``all`` keeps
    every one. Identical sequences yield no spans.
    """
    import difflib

    sm = difflib.SequenceMatcher(None, list(student_ids), list(target_ids), autojunk=False)
    spans: list[tuple[int, int]] = []
    n = len(student_ids)
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        span = (i1, i2) if i2 > i1 else (i1, min(i1 + 1, n))
        if span[0] < span[1]:
            spans.append(span)
        if mode == "first":
            break
    return spans


def narrowed_call_spans(spans, call_placed, hinted_turns, student_ids, encode_fn, mode: str):
    """Mask spans with target-bearing, actually-placed call hints narrowed to the tokens
    the corrected call changes; everything else passes through. A diff that comes back
    empty keeps the full span rather than dropping the hint."""
    if mode == "span":
        return spans
    out = []
    for span, placed, (*_, at, target) in zip(spans, call_placed, hinted_turns, strict=True):
        s, e = span
        if placed and at == "call" and target and e > s:
            subs = divergence_spans(student_ids[s:e].tolist(), encode_fn(target), mode)
            out += [(s + a, s + b) for a, b in subs] or [span]
        else:
            out.append(span)
    return out


def call_target_rows(
    spans, call_placed, hinted_turns, student_ids, encode_fn, response_len: int
):
    """Per-position corrected-token ids for placed, target-bearing call hints, aligned by
    the same token diff as the mask narrowing: position i of a replace block maps to the
    corrected block's token at the same offset; an insert's boundary position maps to the
    first inserted token; everywhere else -1 (no forcing target)."""
    import difflib

    out = torch.full((response_len,), -1, dtype=torch.int64)
    for span, placed, (*_, at, target) in zip(spans, call_placed, hinted_turns, strict=True):
        s, e = span
        if not (placed and at == "call" and target and e > s):
            continue
        t_ids = encode_fn(target)
        sm = difflib.SequenceMatcher(None, student_ids[s:e].tolist(), list(t_ids), autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            if i2 == i1:  # insert: the boundary position should have produced t_ids[j1]
                if i1 < e - s:
                    out[s + i1] = int(t_ids[j1])
                continue
            for k in range(i2 - i1):
                if j1 + k < j2:
                    out[s + i1 + k] = int(t_ids[j1 + k])
    return out


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


def forced_target_spans(orig_ids: list[int], target_ids: list[int], mode: str) -> list[tuple[int, int]]:
    """Supervised positions on the CORRECTED call's own grid (``call_target=forced``).

    Mirrors ``divergence_spans`` but keeps the target side of each differing block: those
    are the positions that exist after the corrected call replaces the student's, so
    insertions are supervised as real tokens rather than a boundary. A ``delete`` (the
    student wrote tokens the correction removes) supervises the single target position
    that follows the removal — the token that must not be the student's deleted one.
    ``span`` covers the whole corrected call.
    """
    import difflib

    n = len(target_ids)
    if mode == "span":
        return [(0, n)] if n else []
    sm = difflib.SequenceMatcher(None, list(orig_ids), list(target_ids), autojunk=False)
    spans: list[tuple[int, int]] = []
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        span = (j1, j2) if j2 > j1 else (j1, min(j1 + 1, n))
        if span[0] < span[1]:
            spans.append(span)
        if mode == "first":
            break
    return spans


def forced_call_swap(
    response_ids: torch.Tensor,
    hinted_turns: list[tuple],
    encode_fn,
    call_open_ids: torch.Tensor,
    call_close_ids: torch.Tensor,
    mode: str,
) -> Optional[dict]:
    """Replace the first target-bearing call hint's failed call with its corrected call.

    Returns None when no trustworthy swap exists: no call hint with a target, the call's
    opening or closing tokens are not found inside the hinted turn (truncated call), or
    the correction tokenizes identically to the student's call. Otherwise a dict:

    - ``response_ids``: the modified row
    - ``at`` / ``removed`` / ``inserted``: splice coordinates for sibling per-position rows
      (``row[:at] + fill*inserted + row[at+removed:]``)
    - ``hinted_turns``: the input hints with spans shifted onto the modified grid
    - ``mask_spans``: supervised (start, end) spans on the modified response grid
    - ``hint_idx``: which hint was swapped (its build span must not be re-narrowed:
      student and target are now identical there, so a diff would keep the full span)
    """
    hint_idx = next(
        (k for k, (*_, at, target) in enumerate(hinted_turns) if at == "call" and target), None
    )
    if hint_idx is None:
        return None
    _, start, end, *_ , target = hinted_turns[hint_idx]
    call_open_ids = call_open_ids.to(response_ids.dtype)
    call_close_ids = call_close_ids.to(response_ids.dtype)
    call_at = _find_subseq(response_ids, call_open_ids, start, end)
    if call_at is None:
        return None
    close_at = _find_subseq(response_ids, call_close_ids, call_at, end)
    if close_at is None:
        return None
    close_end = close_at + call_close_ids.shape[0]
    orig = response_ids[call_at:close_end]
    t_ids = torch.tensor(encode_fn(target), dtype=response_ids.dtype)
    if t_ids.shape[0] == 0 or torch.equal(orig, t_ids):
        return None

    delta = t_ids.shape[0] - orig.shape[0]
    new_response = torch.cat([response_ids[:call_at], t_ids, response_ids[close_end:]])
    adjusted = [
        (step, s + delta if s >= close_end else s, e + delta if e >= close_end else e, *rest)
        for step, s, e, *rest in hinted_turns
    ]
    mask_spans = [
        (call_at + a, call_at + b)
        for a, b in forced_target_spans(orig.tolist(), t_ids.tolist(), mode)
    ]
    return {
        "response_ids": new_response,
        "at": call_at,
        "removed": int(orig.shape[0]),
        "inserted": int(t_ids.shape[0]),
        "hinted_turns": adjusted,
        "mask_spans": mask_spans,
        "hint_idx": hint_idx,
    }


def splice_row(row: torch.Tensor, at: int, removed: int, fill: torch.Tensor) -> torch.Tensor:
    """Per-position sibling of a forced swap: cut ``removed`` positions at ``at``, insert ``fill``."""
    return torch.cat([row[:at], fill.to(row.dtype), row[at + removed:]])


def turn_token_mask(response_len: int, spans: list[tuple[int, int]]) -> torch.Tensor:
    """Per-token distillation mask: 1 on the scored spans, 0 elsewhere."""
    mask = torch.zeros(response_len, dtype=torch.float32)
    for start, end in spans:
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
