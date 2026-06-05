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
from verl.utils.model import compute_position_id_with_mask


def reconstruct_padded_teacher_from_nested(
    teacher_input_ids: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rebuild the SDPO teacher tensors from per-sample no-padding (nested) tensors.

    The transfer-queue path stores the teacher sequence (teacher prompt followed by the
    response), the responses and the response mask as jagged/nested tensors. This recreates
    the exact left/right padded layout produced by the legacy trainer (left-padded teacher
    prompt followed by right-padded response) so that the teacher log-prob computation in
    ``_compute_sdpo_teacher_logps_for_loss`` stays identical to the legacy path. The teacher
    attention mask and position ids are recomputed here (they are fully derived from the
    layout) rather than stored.

    Returns padded ``teacher_input_ids``, ``teacher_attention_mask``,
    ``teacher_position_ids``, ``responses`` and ``response_mask``.
    """
    teacher_seq_list = teacher_input_ids.unbind()
    response_list = responses.unbind()
    response_mask_list = response_mask.unbind()
    batch_size = len(teacher_seq_list)

    response_lens = [r.shape[0] for r in response_list]
    # The teacher sequence is [teacher_prompt, response]; recover the prompt by trimming
    # the response from the tail.
    prompt_lens = [teacher_seq_list[i].shape[0] - response_lens[i] for i in range(batch_size)]
    prompt_list = [teacher_seq_list[i][: prompt_lens[i]] for i in range(batch_size)]
    max_prompt_len = max(prompt_lens)
    max_response_len = max(response_lens)

    device = teacher_input_ids.values().device
    id_dtype = teacher_input_ids.values().dtype
    mask_dtype = response_mask.values().dtype

    teacher_prompt_padded = torch.full((batch_size, max_prompt_len), pad_token_id, device=device, dtype=id_dtype)
    teacher_prompt_mask = torch.zeros((batch_size, max_prompt_len), device=device, dtype=mask_dtype)
    responses_padded = torch.full((batch_size, max_response_len), pad_token_id, device=device, dtype=id_dtype)
    response_mask_padded = torch.zeros((batch_size, max_response_len), device=device, dtype=mask_dtype)

    for i in range(batch_size):
        prompt_len, response_len = prompt_lens[i], response_lens[i]
        # left-pad the teacher prompt (tokenizer.padding_side == "left" in the builder)
        teacher_prompt_padded[i, max_prompt_len - prompt_len :] = prompt_list[i]
        teacher_prompt_mask[i, max_prompt_len - prompt_len :] = 1
        # right-pad the response, mirroring the rollout response layout
        responses_padded[i, :response_len] = response_list[i]
        response_mask_padded[i, :response_len] = response_mask_list[i]

    teacher_input_ids = torch.cat([teacher_prompt_padded, responses_padded], dim=1)
    teacher_attention_mask = torch.cat([teacher_prompt_mask, response_mask_padded], dim=1)
    teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)

    return teacher_input_ids, teacher_attention_mask, teacher_position_ids, responses_padded, response_mask_padded


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
