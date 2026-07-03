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

import contextlib
import functools
import inspect
import json
import os
from contextvars import ContextVar
from typing import Optional

from pydantic import BaseModel

from verl.utils.ray_utils import get_event_loop

_trace_enabled: ContextVar[bool] = ContextVar("_trace_enabled", default=True)
_trace_attributes: ContextVar[dict | None] = ContextVar("_trace_attributes", default=None)

# Per-op Langfuse rendering, registered by the modules defining traced ops (see register_langfuse_op).
_LANGFUSE_OPS: dict[str, dict] = {}


def register_langfuse_op(qualname, *, skip=False, no_io=False, as_type="span", root=False, name=None, output_fn=None):
    """Customize how an ``@rollout_trace_op``-decorated op renders in Langfuse.

    Args:
        skip: Emit no span (e.g. raw token-id I/O with no tokenizer in scope).
        no_io: Keep the span but drop input/output (payloads that stall the UI).
        as_type: Langfuse observation type (default "span").
        root: This op is the trace root; it sets the trace identity.
        name: Span name (default: the qualname).
        output_fn: Maps the op result to the span output.
    """
    _LANGFUSE_OPS[qualname] = {
        "skip": skip,
        "no_io": no_io,
        "as_type": as_type,
        "root": root,
        "name": name or qualname,
        "output_fn": output_fn,
    }


class RolloutTraceConfig:
    """Configuration for rollout tracing with various backends.

    Singleton configuration class for managing rollout trace settings across different
            tracing backends like Weave, MLflow, and Trackio.

    Args:
        backend (Optional[str]): Tracing backend to use ('weave', 'mlflow', or None).
        client (Optional[object]): Client instance for the selected backend.
        token2text (bool): Whether to convert tokens to text in traces. Defaults to False.
        project_name (str): Name of the project for tracing.
        experiment_name (str): Name of the experiment for tracing.
        max_samples_per_step_per_worker (Optional[int]): Maximum number of unique samples to trace
            per worker per step. If None, all samples are traced. If set, each worker will randomly
            select up to this many unique samples to trace (including all their rollouts for GRPO).
            Total traces = max_samples_per_step_per_worker * num_workers * n_rollouts_per_sample.
    """

    _instance: Optional["RolloutTraceConfig"] = None
    backend: str | None = None
    client: object | None = None
    token2text: bool = False
    _initialized: bool = False
    project_name: str = None
    experiment_name: str = None
    max_samples_per_step_per_worker: int | None = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls) -> "RolloutTraceConfig":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def init(
        cls,
        project_name: str,
        experiment_name: str,
        backend: str,
        token2text: bool = False,
        max_samples_per_step_per_worker: int | None = None,
    ):
        config = cls.get_instance()
        if config._initialized:
            return

        config.backend = backend
        config.token2text = token2text
        config.project_name = project_name
        config.experiment_name = experiment_name
        config.max_samples_per_step_per_worker = max_samples_per_step_per_worker

        if backend == "weave":
            import weave

            config.client = weave.init(project_name)
        elif backend == "mlflow":
            import mlflow

            mlflow.config.enable_async_logging()
            config.client = mlflow

            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlruns.db")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            mlflow.set_experiment(project_name)
        elif backend == "trackio":
            import trackio
            from trackio import context_vars

            if context_vars.current_run.get() is None:
                trackio.init(project=project_name, name=experiment_name, config={"framework": "verl"})
            config.client = trackio
        elif backend == "langfuse":
            from langfuse import Langfuse

            # LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST come from env.
            config.client = Langfuse()
        else:
            config.client = None

        config._initialized = True

    @classmethod
    def get_backend(cls) -> str | None:
        return cls.get_instance().backend

    @classmethod
    def get_client(cls) -> object | None:
        return cls.get_instance().client

    @classmethod
    def enable_token2text(cls) -> bool | None:
        return cls.get_instance().token2text

    @classmethod
    def reset(cls):
        cls._instance = None


@contextlib.contextmanager
def rollout_trace_attr(
    sample_index=None, step=None, rollout_n=None, name="rollout_trace", validate=False, trace: bool = True
):
    """A context manager to add attributes to a trace for the configured backend.

    Args:
        sample_index: Sample index for the trace.
        step: Training step number.
        rollout_n: Rollout number (for GRPO with multiple rollouts per sample).
        name: Name for the trace span (used by mlflow backend).
        validate: Whether this is a validation run.
        trace: If False, disables tracing for the duration of the context.
    """
    backend = RolloutTraceConfig.get_backend()

    should_skip = backend is not None and not trace

    if should_skip:
        token = _trace_enabled.set(False)
        try:
            yield
        finally:
            _trace_enabled.reset(token)
        return

    # Build attributes for the trace
    attributes = {}
    if backend:
        if sample_index is not None:
            attributes["sample_index"] = sample_index
        if step is not None:
            attributes["step"] = step
        if rollout_n is not None:
            attributes["rollout_n"] = rollout_n
        attributes["validate"] = validate
        attributes["experiment_name"] = RolloutTraceConfig.get_instance().experiment_name

    if not attributes or backend is None:
        yield
        return

    token = _trace_attributes.set(attributes)
    if backend == "weave":
        import weave

        try:
            with weave.attributes(attributes):
                yield
        finally:
            _trace_attributes.reset(token)
    elif backend == "mlflow":
        import mlflow

        try:
            with mlflow.start_span(name=name) as span:
                trace_id = span.trace_id
                for key, value in attributes.items():
                    mlflow.set_trace_tag(trace_id, str(key), str(value))
                yield
        finally:
            _trace_attributes.reset(token)
    elif backend == "langfuse":
        # no wrapper span: the registered root op sets the trace identity; just stash attrs + flush
        client = RolloutTraceConfig.get_client()
        try:
            yield
        finally:
            try:
                if client is not None:
                    client.flush()
            finally:
                _trace_attributes.reset(token)
    else:
        try:
            yield
        finally:
            _trace_attributes.reset(token)


def _json_trace_content(value):
    if isinstance(value, BaseModel):
        value = value.model_dump()
    return json.dumps(value, default=str, ensure_ascii=False)


def _json_trace_metadata(value):
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(k): _json_trace_metadata(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_trace_metadata(v) for v in value]
    return str(value)


def _trackio_message_dict(message):
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    if not isinstance(role, str):
        return None
    return dict(message)


def _trackio_output_dict(output):
    if isinstance(output, BaseModel):
        return output.model_dump()
    if isinstance(output, dict):
        return output
    if hasattr(output, "__dict__"):
        return dict(vars(output))
    return None


def _trackio_trace_key(op_name):
    return "rollout_trace/" + "".join(char if char.isalnum() or char in "._-" else "_" for char in op_name)


def _trackio_trace_step(attributes):
    step = attributes.get("step")
    if step is None:
        return None
    try:
        return int(step)
    except (TypeError, ValueError):
        return None


def _log_trackio_trace(op_name, inputs, output=None, exception=None):
    trackio = RolloutTraceConfig.get_client()
    attributes = _current_trace_attributes()
    metadata_inputs = {key: value for key, value in inputs.items() if key != "messages"}
    output_dict = _trackio_output_dict(output)
    metadata = {
        "op": op_name,
        "backend": "trackio",
        "experiment_name": RolloutTraceConfig.get_instance().experiment_name,
        "inputs": _json_trace_metadata(metadata_inputs),
        **{key: _json_trace_metadata(value) for key, value in attributes.items()},
    }
    if exception is not None:
        metadata["status"] = "error"
        metadata["exception_type"] = type(exception).__name__
    else:
        metadata["status"] = "success"
        metadata["output"] = _json_trace_metadata(output_dict if output_dict is not None else output)

    messages = []
    input_messages = inputs.get("messages") if isinstance(inputs, dict) else None
    if isinstance(input_messages, list):
        messages = [
            message for message in (_trackio_message_dict(message) for message in input_messages) if message is not None
        ]

    if not messages:
        messages = [
            {"role": "system", "content": f"verl rollout trace operation: {op_name}"},
            {"role": "user", "content": _json_trace_content({"inputs": inputs})},
        ]

    if exception is not None:
        messages.append(
            {
                "role": "assistant",
                "content": _json_trace_content(
                    {
                        "exception_type": type(exception).__name__,
                        "exception": str(exception),
                    }
                ),
            }
        )
    elif output_dict is not None and output_dict.get("response_text"):
        messages.append({"role": "assistant", "content": str(output_dict["response_text"])})
    elif output_dict is not None and output_dict.get("answer"):
        messages.append({"role": "assistant", "content": str(output_dict["answer"])})
    else:
        messages.append({"role": "assistant", "content": _json_trace_content({"output": output})})

    trackio.log(
        {_trackio_trace_key(op_name): trackio.Trace(messages=messages, metadata=metadata)},
        step=_trackio_trace_step(attributes),
    )


def _current_trace_attributes():
    backend = RolloutTraceConfig.get_backend()
    if backend == "weave":
        from weave.trace.context import call_context

        return {**call_context.call_attributes.get()}
    return {**(_trace_attributes.get() or {})}


def _apply_trace_identity(client):
    """Set the trace identity (name/session/metadata/tags) from the stashed rollout attributes."""
    attrs = _trace_attributes.get() or {}
    si = attrs.get("sample_index")
    metadata = {k: v for k, v in attrs.items() if k != "validate"}
    tags = ["validate" if attrs.get("validate") else "train"]
    try:
        client.update_current_trace(
            name="agent_loop",
            session_id=str(si) if si is not None else None,
            metadata=metadata,
            tags=tags,
        )
    except Exception:
        pass


@contextlib.contextmanager
def _langfuse_op_span(cfg, qualname, inputs):
    """Open the langfuse span for a traced op per its registered rendering (shared by both wrappers)."""
    client = RolloutTraceConfig.get_client()
    seg = (_trace_attributes.get() or {}).get("segment_index")
    with client.start_as_current_observation(
        as_type=cfg.get("as_type", "span"),
        name=cfg.get("name", qualname),
        input=None if cfg.get("no_io") else inputs,
        metadata={"segment_index": seg} if seg is not None else None,
    ) as span:
        if cfg.get("root"):
            _apply_trace_identity(client)
        try:
            yield span
        except Exception as e:
            span.update(level="ERROR", status_message=str(e))
            raise


def rollout_trace_set_attr(key, value):
    """Set an attribute on the active rollout trace (surfaces as langfuse span metadata)."""
    attrs = dict(_trace_attributes.get() or {})
    attrs[key] = value
    _trace_attributes.set(attrs)


def rollout_trace_event(name, metadata=None, input=None, output=None):
    """Emit a zero-duration event observation on the active trace (langfuse only)."""
    if RolloutTraceConfig.get_backend() != "langfuse":
        return
    client = RolloutTraceConfig.get_client()
    if client is None:
        return
    try:
        client.create_event(name=name, input=input, output=output, metadata=metadata)
    except Exception:
        pass


def rollout_trace_tool(name, command=None, observation=None, status=None, execution_time=None):
    """Record a decoded tool-call span on the active trace (langfuse only)."""
    if RolloutTraceConfig.get_backend() != "langfuse":
        return
    client = RolloutTraceConfig.get_client()
    if client is None:
        return
    with client.start_as_current_observation(as_type="tool", name=f"tool:{name}", input=command) as span:
        try:
            upd = {"output": observation, "metadata": {"status": status, "execution_time": execution_time}}
            if status is not None and status != "ok":
                upd["level"] = "ERROR"
                upd["status_message"] = str(status)
            span.update(**upd)
        except Exception:
            pass


def rollout_trace_generation(name, model=None, input=None, output=None, usage=None):
    """Record an LLM call as a generation observation with chat I/O and token usage (langfuse only)."""
    if RolloutTraceConfig.get_backend() != "langfuse":
        return
    client = RolloutTraceConfig.get_client()
    if client is None:
        return
    with client.start_as_current_observation(as_type="generation", name=name, model=model, input=input) as gen:
        try:
            upd = {}
            if output is not None:
                upd["output"] = output
            if usage is not None:
                upd["usage_details"] = usage
            if upd:
                gen.update(**upd)
        except Exception:
            pass


def rollout_trace_score(name, value, comment=None, data_type=None):
    """Attach a typed score to the active trace (langfuse only; never raises)."""
    if RolloutTraceConfig.get_backend() != "langfuse":
        return
    client = RolloutTraceConfig.get_client()
    if client is None:
        return
    kw = {"name": name, "value": value}
    if comment is not None:
        kw["comment"] = comment
    if data_type is not None:
        kw["data_type"] = data_type
    try:
        if hasattr(client, "score_current_trace"):
            client.score_current_trace(**kw)
        else:
            client.create_score(trace_id=client.get_current_trace_id(), **kw)
    except Exception:
        pass


def rollout_trace_op(func):
    @functools.wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        if not _trace_enabled.get():
            return await func(self, *args, **kwargs)

        backend = RolloutTraceConfig.get_backend()
        enable_token2text = RolloutTraceConfig.enable_token2text()
        if backend is None:
            return await func(self, *args, **kwargs)

        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        inputs = dict(bound_args.arguments)
        del inputs["self"]

        async def add_token2text(self, result):
            if hasattr(result, "prompt_ids") and hasattr(self, "tokenizer") and hasattr(self.tokenizer, "decode"):
                # Use model_dump() for Pydantic models to get a proper copy,
                # otherwise vars() returns a reference to internal __dict__ which
                # can cause serialization issues with MLflow
                if isinstance(result, BaseModel):
                    _result = result.model_dump()
                else:
                    _result = dict(vars(result))
                loop = get_event_loop()
                if hasattr(result, "prompt_ids"):
                    prompt_text = await loop.run_in_executor(None, self.tokenizer.decode, result.prompt_ids)
                    _result["prompt_text"] = prompt_text

                if hasattr(result, "response_ids"):
                    response_text = await loop.run_in_executor(None, self.tokenizer.decode, result.response_ids)
                    _result["response_text"] = response_text
                return _result
            return result

        if backend == "weave":
            tracer = RolloutTraceConfig.get_client()

            cur_attributes = _current_trace_attributes()
            call = tracer.create_call(op=func.__qualname__, inputs=inputs, attributes=cur_attributes)
            try:
                result = await func(self, *args, **kwargs)

                if enable_token2text:
                    _result = await add_token2text(self, result)
                    tracer.finish_call(call, output=_result)
                else:
                    tracer.finish_call(call, output=result)

                return result

            except Exception as e:
                tracer.finish_call(call, exception=e)
                raise e
        elif backend == "mlflow":
            import mlflow

            with mlflow.start_span(name=func.__qualname__) as span:
                span.set_inputs(inputs)
                result = await func(self, *args, **kwargs)
                if enable_token2text:
                    _result = await add_token2text(self, result)
                    span.set_outputs(_result)
                else:
                    span.set_outputs(result)

            return result
        elif backend == "trackio":
            try:
                result = await func(self, *args, **kwargs)
                if enable_token2text:
                    _result = await add_token2text(self, result)
                    _log_trackio_trace(func.__qualname__, inputs, output=_result)
                else:
                    _log_trackio_trace(func.__qualname__, inputs, output=result)
                return result
            except Exception as e:
                _log_trackio_trace(func.__qualname__, inputs, exception=e)
                raise e

        elif backend == "langfuse":
            cfg = _LANGFUSE_OPS.get(func.__qualname__) or {}
            if cfg.get("skip"):
                return await func(self, *args, **kwargs)
            with _langfuse_op_span(cfg, func.__qualname__, inputs) as span:
                result = await func(self, *args, **kwargs)
                output_fn = cfg.get("output_fn")
                if cfg.get("no_io"):
                    pass
                elif output_fn is not None:
                    span.update(output=output_fn(result))
                elif enable_token2text:
                    span.update(output=await add_token2text(self, result))
                else:
                    span.update(output=result)
                return result

        else:
            return await func(self, *args, **kwargs)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not _trace_enabled.get():
            return func(self, *args, **kwargs)

        backend = RolloutTraceConfig.get_backend()
        if backend is None:
            return func(self, *args, **kwargs)

        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        inputs = dict(bound_args.arguments)
        del inputs["self"]

        if backend == "weave":
            tracer = RolloutTraceConfig.get_client()

            cur_attributes = _current_trace_attributes()
            call = tracer.create_call(op=func.__qualname__, inputs=inputs, attributes=cur_attributes)
            try:
                result = func(self, *args, **kwargs)
                tracer.finish_call(call, output=result)
                return result
            except Exception as e:
                tracer.finish_call(call, exception=e)
                raise e
        elif backend == "mlflow":
            import mlflow

            return mlflow.trace(func)(self, *args, **kwargs)
        elif backend == "trackio":
            try:
                result = func(self, *args, **kwargs)
                _log_trackio_trace(func.__qualname__, inputs, output=result)
                return result
            except Exception as e:
                _log_trackio_trace(func.__qualname__, inputs, exception=e)
                raise e
        elif backend == "langfuse":
            cfg = _LANGFUSE_OPS.get(func.__qualname__) or {}
            if cfg.get("skip"):
                return func(self, *args, **kwargs)
            with _langfuse_op_span(cfg, func.__qualname__, inputs) as span:
                result = func(self, *args, **kwargs)
                output_fn = cfg.get("output_fn")
                if cfg.get("no_io"):
                    pass
                elif output_fn is not None:
                    span.update(output=output_fn(result))
                else:
                    span.update(output=result)
                return result
        else:
            return func(self, *args, **kwargs)

    return async_wrapper if inspect.iscoroutinefunction(func) else wrapper
