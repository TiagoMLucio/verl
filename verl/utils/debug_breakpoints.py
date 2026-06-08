"""Gated breakpoints for the Ray Distributed Debugger VS Code extension.

``maybe_wait_at(name)`` calls the builtin ``breakpoint()`` when ``name`` (or a
group containing it) is listed in ``VERL_RAY_BREAKPOINTS``. The launcher sets it
from its ``RAY_BREAKPOINTS`` arg, e.g. ``RAY_BREAKPOINTS=all`` or
``RAY_BREAKPOINTS=agent,sdpo``.
"""

from __future__ import annotations

import os

GROUPS = {
    "trainer": {"taskrunner", "init_workers", "fit", "actor_init_model", "compute_log_prob", "compute_ref_log_prob"},
    "sdpo": {"sdpo_teacher_loss", "update_actor"},
    "rollout": {"agent_loop", "vllm_server"},
    "agent": {"agent_run", "tool", "reward"},
}
GROUPS["all"] = set().union(*GROUPS.values())
GROUPS["all_but_vllm"] = GROUPS["all"] - {"vllm_server"}


def _enabled() -> set[str]:
    names: set[str] = set()
    for item in os.environ.get("VERL_RAY_BREAKPOINTS", "").replace(",", " ").split():
        item = item.lower()
        names |= GROUPS.get(item, {item})
    return names


def maybe_wait_at(name: str) -> None:
    if name in _enabled():
        breakpoint()
