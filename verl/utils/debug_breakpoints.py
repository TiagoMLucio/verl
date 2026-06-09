"""Gated breakpoints for the Ray Distributed Debugger VS Code extension.

    if should_break("agent_run"): breakpoint()   # stops on THIS line, in your code

Fires when name (or a group containing it) is in VERL_RAY_BREAKPOINTS, set by the
launcher's RAY_BREAKPOINTS arg (e.g. RAY_BREAKPOINTS=all, or agent,sdpo).
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


def should_break(name: str) -> bool:
    return name in _enabled()
