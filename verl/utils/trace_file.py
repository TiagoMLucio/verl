"""Local wall-clock trace sink (Chrome Trace Event Format, one JSONL per process).

Enabled by ``VERL_TRACE_DIR``: every traced span additionally appends an ``X``
(complete) event to ``$VERL_TRACE_DIR/<host>-<pid>.trace.jsonl``, independent of
the rollout-trace backend. Files are merged into a single Perfetto-loadable
trace (slices + hardware counters) by the orchestration repo's build script.

Lanes: ``pid`` is the OS process (trainer / each rollout worker); ``tid`` is a
stable per-rollout id derived from the rollout-trace attributes
(``sample_index``/``rollout_n``), so one row per concurrent rollout and nested
slices for its phases. Trainer-side spans use tid 0.
"""

import json
import os
import socket
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar

__all__ = ["enabled", "emit", "span"]

_lock = threading.Lock()
_file = None
_dir_checked = False
# lane inherited by every span in the same asyncio task tree; concurrent tasks
# without rollout attributes still get distinct rows instead of piling on tid 0
_lane_cv: ContextVar[int | None] = ContextVar("trace_file_lane", default=None)


def enabled() -> bool:
    return bool(os.getenv("VERL_TRACE_DIR"))


def _sink():
    global _file, _dir_checked
    directory = os.getenv("VERL_TRACE_DIR")
    if not directory:
        return None
    if _file is None:
        with _lock:
            if _file is None:
                try:
                    os.makedirs(directory, exist_ok=True)
                    path = os.path.join(directory, f"{socket.gethostname()}-{os.getpid()}.trace.jsonl")
                    handle = open(path, "a", buffering=1)
                    handle.write(
                        json.dumps(
                            {
                                "name": "process_name",
                                "ph": "M",
                                "pid": os.getpid(),
                                "tid": 0,
                                "args": {"name": f"{socket.gethostname()}:{os.getpid()}"},
                            }
                        )
                        + "\n"
                    )
                    _file = handle
                except OSError:
                    if not _dir_checked:
                        _dir_checked = True
                        print(f"trace_file: cannot open sink under {directory!r}; tracing to file disabled")
                    return None
    return _file


def _lane(attrs: dict | None) -> int:
    if attrs and attrs.get("sample_index") is not None:
        # stable row per (sample, rollout_n); offset keeps rollouts clear of tid 0
        return 10 + int(attrs["sample_index"]) * 8 + int(attrs.get("rollout_n") or 0) % 8
    inherited = _lane_cv.get()
    if inherited is not None:
        return inherited
    try:
        import asyncio

        task = asyncio.current_task()
    except RuntimeError:
        task = None
    if task is None:
        return 0  # sync context (trainer): sequential, tid 0 is safe
    return 1_000_000 + id(task) % 1_000_000


def emit(name: str, t_start: float, t_end: float, attrs: dict | None = None, **args) -> None:
    sink = _sink()
    if sink is None:
        return
    if attrs:
        args = {**{k: v for k, v in attrs.items() if k in ("sample_index", "step", "rollout_n", "validate")}, **args}
    event = {
        "name": name,
        "ph": "X",
        "ts": int(t_start * 1e6),
        "dur": max(1, int((t_end - t_start) * 1e6)),
        "pid": os.getpid(),
        "tid": _lane(attrs),
        "args": args,
    }
    with _lock:
        sink.write(json.dumps(event, default=str) + "\n")


@contextmanager
def span(name: str, attrs_getter=None, **args):
    """Time a block; ``attrs_getter`` is called at exit (context may be set inside)."""
    if not enabled():
        yield
        return
    t_start = time.time()
    # bind the lane at entry so every nested span (and thread hop) inherits the row
    attrs_in = None
    if attrs_getter is not None:
        try:
            attrs_in = attrs_getter()
        except Exception:
            attrs_in = None
    token = _lane_cv.set(_lane(attrs_in))
    error = None
    try:
        yield
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        attrs = None
        if attrs_getter is not None:
            try:
                attrs = attrs_getter()
            except Exception:
                attrs = None
        if error is not None:
            args = {**args, "error": error}
        emit(name, t_start, time.time(), attrs=attrs, **args)
        _lane_cv.reset(token)
