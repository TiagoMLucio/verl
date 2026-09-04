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
"""Reflector hints paired with the turns they land on, and their chat-template rendering."""

from typing import NamedTuple, Optional

import torch

__all__ = [
    "HintedTurn",
    "assistant_header_ids",
    "hint_token_ids",
    "hint_user_turn_ids",
    "select_hinted_turns",
]


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
    """Pair a sample's turn spans with its hints. The rollout ships the
    placement as an optional third element of the ``turn_hints`` entry; a fourth element
    (the ``target`` field older rollout dumps still carry) is ignored.

    Spans are clamped to the (possibly truncated) response; with a cap, the first
    ``max_hinted_turns`` turns are kept (earliest, before the trajectory loses coherence).
    """
    hint_by_step = {int(entry[0]): (entry[1], entry[2] if len(entry) > 2 else "turn")
                    for entry in (extra_fields.get("turn_hints") or [])}
    hinted = []
    for step, start, end in extra_fields.get("turn_spans") or []:
        step, start, end = int(step), int(start), min(int(end), response_len)
        if step in hint_by_step and start < end:
            text, placement = hint_by_step[step]
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
    (``cfg.call_hint_template`` or ``cfg.turn_hint_template``), rendered as a user turn."""
    template = cfg.call_hint_template if hint.is_call else cfg.turn_hint_template
    return torch.tensor(
        hint_user_turn_ids(tokenizer, template.format(hint=hint.text), template_kwargs=template_kwargs),
        dtype=torch.int64,
    )
