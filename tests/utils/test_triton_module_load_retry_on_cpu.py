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
"""A Triton kernel's module is loaded with a driver call that PyTorch's own free-and-retry never sees: with the caching
allocator holding the whole device, the first launch of an unseen specialization (a new autotune key) fails with
"Triton Error [CUDA]: out of memory". The load is retried once after the cache is emptied. A stand-in for Triton's
CompiledKernel, since the load itself needs a GPU."""

import sys
import types

import pytest

from verl.utils import memory_utils

OOM = "Triton Error [CUDA]: out of memory"


class Device:
    def __init__(self):
        self.emptied = 0

    def empty_cache(self):
        self.emptied += 1


@pytest.fixture
def kernel(monkeypatch):
    """A CompiledKernel whose module load fails as scripted: ``failures`` is the list of errors to raise, in order."""

    class CompiledKernel:
        failures = []
        loads = 0

        def _init_handles(self):
            CompiledKernel.loads += 1
            if CompiledKernel.failures:
                raise CompiledKernel.failures.pop(0)
            return "loaded"

    compiler = types.ModuleType("triton.compiler.compiler")
    compiler.CompiledKernel = CompiledKernel
    package = types.ModuleType("triton.compiler")
    package.compiler = compiler
    triton = types.ModuleType("triton")
    triton.compiler = package
    for name, module in (("triton", triton), ("triton.compiler", package), ("triton.compiler.compiler", compiler)):
        monkeypatch.setitem(sys.modules, name, module)
    device = Device()
    monkeypatch.setattr(memory_utils, "get_torch_device", lambda: device)
    memory_utils.retry_triton_module_loads()
    return CompiledKernel, device


def test_a_module_load_that_runs_out_of_memory_is_retried_after_the_cache_is_emptied(kernel):
    compiled, device = kernel
    compiled.failures = [RuntimeError(OOM)]
    assert compiled()._init_handles() == "loaded"
    assert (compiled.loads, device.emptied) == (2, 1)


def test_a_load_that_fails_again_is_the_callers_error(kernel):
    compiled, device = kernel
    compiled.failures = [RuntimeError(OOM), RuntimeError(OOM)]
    with pytest.raises(RuntimeError, match="out of memory"):
        compiled()._init_handles()
    assert (compiled.loads, device.emptied) == (2, 1)


def test_other_errors_pass_through_untouched(kernel):
    compiled, device = kernel
    compiled.failures = [RuntimeError("Triton Error [CUDA]: invalid argument")]
    with pytest.raises(RuntimeError, match="invalid argument"):
        compiled()._init_handles()
    assert (compiled.loads, device.emptied) == (1, 0)


def test_a_load_that_works_never_touches_the_cache(kernel):
    compiled, device = kernel
    assert compiled()._init_handles() == "loaded"
    assert (compiled.loads, device.emptied) == (1, 0)


def test_installing_twice_wraps_once(kernel):
    compiled, device = kernel
    memory_utils.retry_triton_module_loads()
    compiled.failures = [RuntimeError(OOM)]
    assert compiled()._init_handles() == "loaded"
    assert (compiled.loads, device.emptied) == (2, 1)


def test_without_triton_nothing_is_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton.compiler.compiler", None)
    memory_utils.retry_triton_module_loads()
