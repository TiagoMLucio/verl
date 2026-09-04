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

from verl.trainer.ppo.sdpo.teacher_meta import SubRowSpan, body_slices, unpack
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
    # presence, not loss, mask: the teacher must attend to tool-observation tokens that response_mask zeroes
    response_presence_padded = torch.zeros((batch_size, max_response_len), device=device, dtype=mask_dtype)

    for i in range(batch_size):
        prompt_len, response_len = prompt_lens[i], response_lens[i]
        # left-pad the teacher prompt (tokenizer.padding_side == "left" in the builder)
        teacher_prompt_padded[i, max_prompt_len - prompt_len :] = prompt_list[i]
        teacher_prompt_mask[i, max_prompt_len - prompt_len :] = 1
        # right-pad the response, mirroring the rollout response layout
        responses_padded[i, :response_len] = response_list[i]
        response_mask_padded[i, :response_len] = response_mask_list[i]
        response_presence_padded[i, :response_len] = 1

    teacher_input_ids = torch.cat([teacher_prompt_padded, responses_padded], dim=1)
    teacher_attention_mask = torch.cat([teacher_prompt_mask, response_presence_padded], dim=1)
    teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)

    return teacher_input_ids, teacher_attention_mask, teacher_position_ids, responses_padded, response_mask_padded


def explode_turn_teacher_rows(
    teacher_input_ids: torch.Tensor,
    teacher_seq_meta: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[SubRowSpan]]:
    """Unpack per-sample spliced teacher rows into the sub-batch the teacher forward scores.

    ``teacher_seq_meta`` is flat per sample (see :mod:`verl.trainer.ppo.sdpo.teacher_meta`).
    Each sub-row's body is its verbatim tail, so the padded reconstruction contract holds.
    One sub-row per hinted turn, each carrying only its own hint, so the teacher never scores
    a turn from a state where its earlier advice was visibly ignored. Un-hinted rows ship a
    degenerate 2-token sub-row: the teacher forward must run on every rank every micro-batch
    so its dp-group collectives stay in lockstep. Returns nested sequences / bodies / body
    masks plus one :class:`SubRowSpan` per sub-row.
    """
    seq_list = teacher_input_ids.unbind()
    meta_list = teacher_seq_meta.unbind()
    mask_list = response_mask.unbind()

    sub_seqs, sub_resps, sub_masks, sub_spans = [], [], [], []
    resp_list = responses.unbind()
    assert len(seq_list) == len(resp_list), (
        f"teacher rows ({len(seq_list)}) and responses ({len(resp_list)}) disagree on batch size"
    )
    for i, (seq, meta) in enumerate(zip(seq_list, meta_list, strict=True)):
        offset = 0
        for j, sub_row in enumerate(unpack(meta.tolist(), sample=i)):
            assert 0 < sub_row.body_len <= sub_row.total_len and offset + sub_row.total_len <= seq.shape[0], (
                f"teacher_seq_meta malformed for sample {i} sub-row {j}: total {sub_row.total_len}, "
                f"body {sub_row.body_len}, offset {offset}, seq len {seq.shape[0]}"
            )
            assert sub_row.end <= resp_list[i].shape[0], (
                f"teacher_seq_meta span exceeds the response row for sample {i}: "
                f"end {sub_row.end} > response len {resp_list[i].shape[0]}"
            )
            sub = seq[offset : offset + sub_row.total_len]
            sub_seqs.append(sub)
            sub_resps.append(sub[-sub_row.body_len :])
            sub_masks.append(mask_list[i].new_ones(sub_row.body_len))
            sub_spans.append(SubRowSpan(i, sub_row.body_start, sub_row.start, sub_row.end))
            offset += sub_row.total_len

    return (
        torch.nested.nested_tensor(sub_seqs, layout=torch.jagged),
        torch.nested.nested_tensor(sub_resps, layout=torch.jagged),
        torch.nested.nested_tensor(sub_masks, layout=torch.jagged),
        sub_spans,
    )


def _keep_positions(prefix_lens: torch.Tensor, spans_per_row: list[list[tuple[int, int]]]) -> torch.Tensor:
    """Row-relative logits positions for (offset, length) spans past each row's prefix,
    shifted -1 because position k predicts token k+1 (matches no_padding_2_padding).
    Span-less rows keep one dummy position so an all-unhinted micro-batch still yields
    a graph-connected (zero-contribution) loss."""
    rows = []
    for prefix_len, row_spans in zip(prefix_lens.tolist(), spans_per_row, strict=True):
        if row_spans:
            rows.append(
                torch.cat(
                    [torch.arange(prefix_len + off - 1, prefix_len + off - 1 + length) for off, length in row_spans]
                )
            )
        else:
            rows.append(torch.tensor([prefix_len - 1], dtype=torch.int64))
    return torch.nested.nested_tensor(rows, layout=torch.jagged)


def turn_keep_positions(sub_seqs: torch.Tensor, sub_resps: torch.Tensor, sub_spans: list[SubRowSpan]) -> torch.Tensor:
    """Logits positions scoring the hinted spans on the spliced teacher rows."""
    prefix_lens = sub_seqs.offsets().diff() - sub_resps.offsets().diff()
    spans = [[(body.start, body.stop - body.start)] for _, _, body, _ in body_slices(sub_spans)]
    return _keep_positions(prefix_lens, spans)


def response_keep_positions(
    input_ids: torch.Tensor, responses: torch.Tensor, teacher_seq_meta: torch.Tensor
) -> torch.Tensor:
    """Logits positions scoring the hinted spans on the student's own prompt+response rows."""
    prefix_lens = input_ids.offsets().diff() - responses.offsets().diff()
    # the student scores every hinted span in one pass, so it keeps their union
    spans = [
        [(sub_row.start, sub_row.end - sub_row.start) for sub_row in unpack(meta.tolist())]
        for meta in teacher_seq_meta.unbind()
    ]
    return _keep_positions(prefix_lens, spans)


def attach_response_keep_positions(data) -> None:
    """Mark the SDPO update pass for span-only logits when turn-mode teacher meta is present."""
    turn_meta = data.get("teacher_seq_meta", None)
    if turn_meta is not None and turn_meta.is_nested:
        data["logits_keep_positions"] = response_keep_positions(data["input_ids"], data["responses"], turn_meta)


def scatter_turn_teacher_outputs(
    sub_outputs: torch.Tensor,
    sub_spans: list[SubRowSpan],
    batch_size: int,
    response_length: int,
) -> torch.Tensor:
    """Scatter padded spliced-body teacher outputs back onto the (batch, response) grid.

    Positions outside the scored spans stay zero; the per-token distillation mask excludes them.
    """
    full = sub_outputs.new_zeros((batch_size, response_length, *sub_outputs.shape[2:]))
    for j, parent, body_slice, response_slice in body_slices(sub_spans):
        if parent >= batch_size or response_slice.stop > response_length or sub_outputs.dim() < 2:
            raise RuntimeError(
                "scatter_turn_teacher_outputs misalignment: "
                f"row j={j} span={sub_spans[j]} vs "
                f"grid=({batch_size},{response_length}) sub_outputs={tuple(sub_outputs.shape)} "
                f"n_sub={len(sub_spans)} all_spans={sub_spans}"
            )
        full[parent, response_slice] = sub_outputs[j, body_slice]
    return full


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
