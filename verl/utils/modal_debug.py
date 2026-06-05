"""Small debugpy pause hooks for Modal/Ray debugging.

Usage — insert a call at the point you want to pause::

    from verl.utils.modal_debug import maybe_wait_at
    maybe_wait_at("fit")

The call is a no-op unless the corresponding VERL_RAY_BREAK_* env var is set
to 1/true/yes/on in the Ray runtime environment.
"""

from __future__ import annotations

import os
import threading

# name -> (env_var, debugpy_port, label)
# Breakpoint names that share a port share a single debugpy listener, but each
# name still only fires once per process lifetime.
BREAKPOINTS = {
    # Port 5684 — TaskRunner / RayPPOTrainer Ray actor
    "taskrunner":           ("VERL_RAY_BREAK_TASKRUNNER",            5684, "TaskRunner.run"),
    "init_workers":         ("VERL_RAY_BREAK_INIT_WORKERS",          5684, "RayPPOTrainer.init_workers"),
    "fit":                  ("VERL_RAY_BREAK_FIT",                   5684, "RayPPOTrainer.fit"),
    # Port 5685 — ActorRolloutRefWorker Ray actor (training + inference)
    "actor_init_model":     ("VERL_RAY_BREAK_ACTOR_INIT_MODEL",      5685, "ActorRolloutRefWorker.init_model"),
    "compute_log_prob":     ("VERL_RAY_BREAK_COMPUTE_LOG_PROB",      5685, "ActorRolloutRefWorker.compute_log_prob"),
    "compute_ref_log_prob": ("VERL_RAY_BREAK_COMPUTE_REF_LOG_PROB",  5685, "ActorRolloutRefWorker.compute_ref_log_prob"),
    "sdpo_teacher_loss":  ("VERL_RAY_BREAK_SDPO_TEACHER_LOSS",       5685, "ActorRolloutRefWorker.sdpo_teacher_loss"),
    "update_actor":         ("VERL_RAY_BREAK_UPDATE_ACTOR",          5685, "ActorRolloutRefWorker.update_actor"),
    # Port 5686 — vLLMHttpServer Ray actor (separate process from actor worker)
    "vllm_server":          ("VERL_RAY_BREAK_VLLM_SERVER",          5686, "vLLMHttpServer.generate"),
    # Port 5687 — AgentLoopWorkerTQ Ray actor (main_ppo_sync rollout agent worker)
    "agent_loop":           ("VERL_RAY_BREAK_AGENT_LOOP",           5687, "AgentLoopWorkerTQ.generate_sequences"),
}

_LOCK = threading.Lock()
_WAITED: set[str] = set()
_LISTENING_PORTS: set[int] = set()


def maybe_wait_at(name: str) -> None:
    """Pause the current Ray process and wait for a VS Code debugpy attach.

    Does nothing when the breakpoint's env var is not set, or when this name
    has already been waited on in this process (one-shot per process).
    """
    config = BREAKPOINTS.get(name)
    if config is None:
        return

    env_var, port, label = config
    if os.environ.get(env_var, "").strip().lower() not in {"1", "true", "yes", "on"}:
        return

    with _LOCK:
        if name in _WAITED:
            return
        _WAITED.add(name)

        try:
            import debugpy
        except Exception as exc:
            print(f"{label} debugpy requested, but debugpy is unavailable: {exc}", flush=True)
            return

        if port not in _LISTENING_PORTS:
            debugpy.listen(("0.0.0.0", port))
            _LISTENING_PORTS.add(port)

        print(f"{label} debugpy is listening on 0.0.0.0:{port}", flush=True)
        print(f"{label} waiting for VS Code attach...", flush=True)

    debugpy.wait_for_client()
    debugpy.breakpoint()
