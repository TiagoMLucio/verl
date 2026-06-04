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

from types import SimpleNamespace

import torch

from verl.utils import tensordict_utils as tu


def has_non_empty_multi_modal_inputs(data) -> bool:
    multi_modal_inputs = tu.get(data, "multi_modal_inputs", default=None)
    if multi_modal_inputs is None:
        return False
    for inputs in multi_modal_inputs:
        if inputs is None:
            continue
        inputs = getattr(inputs, "data", inputs)
        if isinstance(inputs, dict):
            if not inputs:
                continue
            for value in inputs.values():
                if value is None:
                    continue
                if isinstance(value, torch.Tensor) and value.numel() == 0:
                    continue
                return True
        else:
            return True
    return False


class TrustRegionTeacher(torch.nn.Module):
    """Blends ref and student logits for trust-region teacher regularization."""

    def __init__(self, ref_module: torch.nn.Module, student_module: torch.nn.Module, mix_coef: float):
        super().__init__()
        self.ref_module = ref_module
        self.student_module = student_module
        self.mix_coef = float(mix_coef)
        if not 0.0 <= self.mix_coef <= 1.0:
            raise ValueError(f"mix_coef must be in [0,1], got {self.mix_coef}")

    @staticmethod
    def _extract_logits(output) -> torch.Tensor:
        if hasattr(output, "logits"):
            return output.logits
        if isinstance(output, tuple):
            return output[0]
        if isinstance(output, dict):
            return output["logits"]
        raise ValueError(f"Unsupported model output type for trust-region teacher: {type(output)}")

    def forward(self, *args, **kwargs):
        ref_output = self.ref_module(*args, **kwargs)
        student_output = self.student_module(*args, **kwargs)
        ref_logits = self._extract_logits(ref_output)
        student_logits = self._extract_logits(student_output)
        logits = torch.lerp(ref_logits, student_logits, self.mix_coef)
        return SimpleNamespace(logits=logits)
