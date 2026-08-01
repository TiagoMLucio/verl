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

__all__ = ["enabled", "emit", "span"]

_lock = threading.Lock()
_file = None
_dir_checked = False


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
    if not attrs:
        return 0
    sample = attrs.get("sample_index")
    if sample is None:
        return 0
    # stable row per (sample, rollout_n); offset keeps rollouts clear of tid 0
    return 10 + int(sample) * 8 + int(attrs.get("rollout_n") or 0) % 8


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
