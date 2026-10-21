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
"""Worker-side half of :mod:`verl.trainer.ppo.sdpo`: re-pad the nested teacher tensors, split
spliced rows into their base and the branches off it, mark the span-only logits positions, scatter
the teacher outputs back to the response grid, and mix the trust-region teacher."""

from types import SimpleNamespace
from typing import NamedTuple, Optional

import torch
from transformers import DynamicCache

from verl.trainer.ppo.sdpo.teacher_meta import SubRowSpan, unpack, unpack_row
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_name
from verl.utils.model import compute_position_id_with_mask
from verl.workers.utils.padding import left_right_2_no_padding


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


class Branch(NamedTuple):
    """One sub-row as a continuation of its sample's base: ``ids`` (the base token before the split, the
    note, then the base up to the scored span's end) follow the first ``point`` base tokens, at positions
    ``point`` on. The base token is read again so that a sub-row with no note scores its first token too."""

    point: int
    ids: torch.Tensor
    span: SubRowSpan


def teacher_branches(
    teacher_input_ids: torch.Tensor, teacher_seq_meta: torch.Tensor, responses: torch.Tensor
) -> list[tuple[torch.Tensor, list[Branch]]]:
    """Per sample, its base and its sub-rows as branches off it, in the order they leave the base.

    ``teacher_seq_meta`` is flat per sample (see :mod:`verl.trainer.ppo.sdpo.teacher_meta`). One sub-row
    per supervised span, each with only its own note, so the teacher scores every span from the context
    built for it; the sub-rows share the base before their split, so the teacher reads it once. Rows
    without supervised spans ship a 2-token stub: the teacher forward must run on every rank every
    micro-batch so its dp-group collectives stay in lockstep. A sample's contrast sub-rows follow its
    own, flagged ``contrast`` in their span, and its no-note sub-rows follow those, flagged ``anchor``.
    """
    rows = []
    samples = zip(teacher_input_ids.unbind(), teacher_seq_meta.unbind(), responses.unbind(), strict=True)
    for i, (seq, meta, response) in enumerate(samples):
        base, offset, noted = unpack_row(seq, meta.tolist(), sample=i)
        branches = []
        for j, (sub_row, note, contrast, anchor) in enumerate(noted):
            split, stop = sub_row.split, offset + sub_row.end
            assert 0 < split <= offset + sub_row.start < stop <= base.shape[0] and sub_row.end <= response.shape[0], (
                f"teacher_seq_meta malformed for sample {i} sub-row {j}: {sub_row}, response offset {offset}, "
                f"base {base.shape[0]} tokens, response {response.shape[0]}"
            )
            ids = torch.cat([base[split - 1 : split], note, base[split:stop]])
            body_start = note.shape[0] + offset + sub_row.start - split
            branches.append(
                Branch(split - 1, ids, SubRowSpan(i, body_start, sub_row.start, sub_row.end, contrast, anchor))
            )
        rows.append((base, sorted(branches, key=lambda branch: branch.point)))
    return rows


# a forward that continues a cache gets a dense (query x key) SDPA mask, so a long read on goes in pieces
CONTINUATION_PIECE = 2048


def continuation_pieces(start: int, stop: int) -> list[tuple[int, int]]:
    """``[start, stop)`` in pieces of one to two :data:`CONTINUATION_PIECE` tokens, or whole when shorter."""
    n = max(1, (stop - start) // CONTINUATION_PIECE)
    bounds = [start + (stop - start) * k // n for k in range(n + 1)]
    return list(zip(bounds[:-1], bounds[1:], strict=True))


class TeacherCache(DynamicCache):
    """The cache a teacher row's reads and branches continue. Its recurrent (linear-attention) layers keep their
    state in the dtype the kernel returns it in: transformers sizes it by the conv state's, the model's dtype,
    which would round it to bf16 at every handover between forwards, where a single forward keeps it fp32."""

    def update_recurrent_state(self, recurrent_states: torch.Tensor, layer_idx: int, **kwargs) -> torch.Tensor:
        layer = self.layers[layer_idx]
        if not layer.is_recurrent_states_initialized:
            layer.recurrent_states = torch.zeros_like(recurrent_states)
            layer.is_recurrent_states_initialized = True
        return super().update_recurrent_state(recurrent_states, layer_idx, **kwargs)


def recurrent_states(cache) -> list:
    """Copies of a transformers cache's recurrent-layer states, which a forward overwrites in place
    (attention layers append their keys and values instead, so a crop rewinds them)."""
    return [
        (layer.conv_states.clone(), layer.recurrent_states.clone())
        if getattr(layer, "has_previous_state", False)
        else None
        for layer in cache.layers
    ]


def rewind(cache, length: int, states: list) -> None:
    """Put ``cache`` back to the ``length`` tokens it held when ``states`` was taken."""
    cache.crop(length)
    for layer, saved in zip(cache.layers, states, strict=True):
        if saved is not None:
            layer.conv_states.copy_(saved[0])
            layer.recurrent_states.copy_(saved[1])


def _keep_positions(prefix_lens: torch.Tensor, spans_per_row: list[list[tuple[int, int]]]) -> torch.Tensor:
    """Row-relative logits positions for (offset, length) spans past each row's prefix,
    shifted -1 because position k predicts token k+1 (matches no_padding_2_padding).
    Span-less rows keep one dummy position so a micro-batch without supervised spans still yields
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


def response_keep_positions(
    input_ids: torch.Tensor,
    responses: torch.Tensor,
    teacher_seq_meta: torch.Tensor,
    outcome_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Logits positions scoring the supervised spans on the student's own prompt+response rows, and
    with ``outcome_mask`` the outcome term's positions too."""
    prefix_lens = input_ids.offsets().diff() - responses.offsets().diff()
    # the student scores every supervised span in one pass, so it keeps their union
    spans = [
        [(sub_row.start, sub_row.end - sub_row.start) for sub_row in unpack(meta.tolist())]
        for meta in teacher_seq_meta.unbind()
    ]
    if outcome_mask is not None:
        for i, row in enumerate(outcome_mask.unbind()):
            extra = torch.nonzero(row).flatten()
            if extra.numel():
                positions = torch.unique(torch.cat([extra, *(torch.arange(s, s + n) for s, n in spans[i])]))
                cuts = (torch.nonzero(positions.diff() != 1).flatten() + 1).tolist()
                bounds = list(zip([0, *cuts], [*cuts, positions.numel()], strict=True))
                spans[i] = [(int(positions[a]), b - a) for a, b in bounds]
    return _keep_positions(prefix_lens, spans)


def supervised_spans(data) -> list[tuple[int, int, int, int]]:
    """``(row_id, row, start, end)`` for every span the spliced teacher rows score.

    Empty unless the batch carries both the spliced meta and the trainer's row ids, which is
    what ties an exported span back to the trajectory it came from.
    """
    turn_meta = tu.get(data, "teacher_seq_meta", default=None)
    row_ids = tu.get(data, "row_id", default=None)
    if turn_meta is None or not turn_meta.is_nested or row_ids is None:
        return []
    row_ids = (row_ids.to_padded_tensor(-1) if row_ids.is_nested else row_ids).reshape(-1).tolist()
    return [
        (row_id, row, sub_row.start, sub_row.end)
        for row, (row_id, meta) in enumerate(zip(row_ids, turn_meta.unbind(), strict=True))
        for sub_row in unpack(meta.tolist(), row)
    ]


def attach_response_keep_positions(data) -> None:
    """Mark the SDPO update pass for span-only logits when spliced teacher meta is present."""
    turn_meta = data.get("teacher_seq_meta", None)
    if turn_meta is not None and turn_meta.is_nested:
        outcome_mask = data.get("outcome_mask", None)
        data["logits_keep_positions"] = response_keep_positions(
            data["input_ids"], data["responses"], turn_meta, outcome_mask if torch.is_tensor(outcome_mask) else None
        )


# Fields off the response grid that a student-row cut must leave whole: the prompt, the spliced
# teacher rows (every sub-row already ends at its own scored span) and the per-row weights the
# trajectory means divide by.
_TRUNCATION_EXEMPT = frozenset({"prompts", "teacher_input_ids", "teacher_seq_meta", "trace_weight", "outcome_weight"})


def _raise_if_partly_on_the_grid(data, key, lens, resp_lens, seq_lens) -> None:
    """A field that matches the response (or sequence) grid on some rows but not all is an upstream
    length disagreement, not another grid: skipping it would cut the student rows and leave it whole."""
    # a per-row constant (a length-1 placeholder) is on no grid, even where a 1-token pad row matches it
    if len(set(lens)) == 1:
        return
    for grid, ref in (("response", resp_lens), ("sequence", seq_lens)):
        off = [i for i, (n, r) in enumerate(zip(lens, ref, strict=True)) if n != r]
        if len(off) == len(lens):
            continue

        def row_lens(name, i):
            value = data.get(name, None)
            return value.offsets().diff().tolist()[i] if torch.is_tensor(value) and value.is_nested else None

        def row_id(name, i):
            value = data.get(name, None)
            return int(value.reshape(-1)[i]) if torch.is_tensor(value) and not value.is_nested else None

        rows = [
            f"row {i}: {key} {lens[i]}, responses {resp_lens[i]}, response_mask {row_lens('response_mask', i)}, "
            f"input_ids {seq_lens[i]}, row_id {row_id('row_id', i)}, traj_id {row_id('traj_id', i)}"
            for i in off[:8]
        ]
        raise RuntimeError(
            f"SDPO row truncation: {key!r} is on the {grid} grid for {len(lens) - len(off)} of {len(lens)} rows "
            f"but not for {len(off)}; cutting the others would leave it misaligned with the student.\n  "
            + "\n  ".join(rows)
        )


def truncate_rows_after_last_supervised_token(data) -> float:
    """Cut every student row after its last supervised token; returns the kept token fraction.

    Attention is causal and only the hinted spans carry loss, so a position past a row's last
    supervised token reaches no position that does: the logits, the loss and the gradient at every
    scored position are what the whole row would have given. Rows with no supervised token, and
    rows already ending on one, come out untouched.
    """
    mask, meta = data.get("self_distillation_mask", None), data.get("teacher_seq_meta", None)
    if mask is None or meta is None or not mask.is_nested or not meta.is_nested:
        return 1.0

    resp_lens = data["responses"].offsets().diff().tolist()
    seq_lens = data["input_ids"].offsets().diff().tolist()
    keep_lens = []
    for i, (row, resp_len, meta_row) in enumerate(zip(mask.unbind(), resp_lens, meta.unbind(), strict=True)):
        supervised = torch.nonzero(row).flatten()
        if supervised.numel() == 0:
            keep_lens.append(resp_len)
            continue
        keep = int(supervised[-1]) + 1
        span_end = max(sub_row.end for sub_row in unpack(meta_row.tolist(), sample=i))
        # scatter_turn_teacher_outputs writes each sub-row onto [start, end) of the kept grid
        assert span_end == keep, (
            f"row {i}: the supervised mask ends at {keep} but the teacher spans end at {span_end}; "
            "truncating here would drop scored positions"
        )
        keep_lens.append(min(keep, resp_len))

    if keep_lens == resp_lens:
        return 1.0
    prefix_lens = [seq_len - resp_len for seq_len, resp_len in zip(seq_lens, resp_lens, strict=True)]
    seq_keep_lens = [prefix + keep for prefix, keep in zip(prefix_lens, keep_lens, strict=True)]

    for key in list(data.keys()):
        if key in _TRUNCATION_EXEMPT:
            continue
        value = data.get(key)
        if not torch.is_tensor(value) or not value.is_nested:
            continue
        lens = value.offsets().diff().tolist()
        if lens == resp_lens:
            new_lens = keep_lens
        elif lens == seq_lens:
            new_lens = seq_keep_lens
        else:
            _raise_if_partly_on_the_grid(data, key, lens, resp_lens, seq_lens)
            continue
        if value.dim() > 2:
            raise RuntimeError(f"SDPO row truncation does not support the {value.dim()}d field {key!r}")
        data[key] = torch.nested.nested_tensor(
            [row[:n] for row, n in zip(value.unbind(), new_lens, strict=True)], layout=torch.jagged
        )
    return sum(keep_lens) / max(sum(resp_lens), 1)


def scatter_turn_teacher_outputs(
    span_outputs: list[torch.Tensor],
    sub_spans: list[SubRowSpan],
    batch_size: int,
    response_length: int,
    contrast: bool = False,
    anchor: bool = False,
) -> torch.Tensor:
    """Scatter the teacher's outputs on each sub-row's scored span back onto the (batch, response) grid:
    the hint sub-rows, or with ``contrast`` the contrast teacher's, or with ``anchor`` the no-note ones.

    Positions outside the scored spans stay zero; the per-token distillation mask excludes them.
    """
    full = span_outputs[0].new_zeros((batch_size, response_length, *span_outputs[0].shape[1:]))
    for j, (outputs, span) in enumerate(zip(span_outputs, sub_spans, strict=True)):
        if (span.contrast, span.anchor) != (contrast, anchor):
            continue
        if span.parent >= batch_size or span.end > response_length or outputs.shape[0] != span.end - span.start:
            raise RuntimeError(
                "scatter_turn_teacher_outputs misalignment: "
                f"row j={j} span={span} vs grid=({batch_size},{response_length}) outputs={tuple(outputs.shape)} "
                f"n_sub={len(sub_spans)} all_spans={sub_spans}"
            )
        full[span.parent, span.start : span.end] = outputs
    return full


def score_teacher_branches(
    engine, data, student_topk_indices, output_keys, non_tensor_dict, default_keys, loss_function
) -> dict[str, torch.Tensor]:
    """Teacher targets of spliced rows: every sub-row is a branch off its sample's base, which the
    teacher reads once through a cache and continues per branch from its split, scoring the branch's
    span (see :func:`teacher_branches`)."""
    rows = teacher_branches(data["teacher_input_ids"], data["teacher_seq_meta"], data["responses"])
    has_contrast = any(branch.span.contrast for _, branches in rows for branch in branches)
    has_anchor = any(branch.span.anchor for _, branches in rows for branch in branches)
    if (has_contrast or has_anchor) and student_topk_indices is None:
        raise ValueError("SDPO contrast sub-rows need top-k distillation (distillation_topk set).")

    # a sample's forwards in order: read its base on to each split (in pieces once a cache holds some of it),
    # then the branches leaving it there
    steps = []
    for i, (base, branches) in enumerate(rows):
        read = 0
        for branch in branches:
            if branch.point > read:
                pieces = [(0, branch.point)] if read == 0 else continuation_pieces(read, branch.point)
                steps += [(lo, base[lo:hi], None) for lo, hi in pieces]
                read = branch.point
            steps.append((branch.point, branch.ids, branch))
    n_calls = len(steps)
    if torch.distributed.is_initialized():
        # each infer_batch runs FSDP all-gathers and an all-reduce, so every rank makes the same number of calls;
        # a rank with fewer forwards a 2-token stub for the rest; a rank whose rows are all un-hinted still returns
        # the contrast and anchor outputs (zeros) when another rank has them, so every rank's loss reports the same
        # metric keys
        group = engine.get_data_parallel_group()
        device = "cpu" if torch.distributed.get_backend(group) == "gloo" else get_device_name()
        count = torch.tensor([n_calls, int(has_contrast), int(has_anchor)], device=device)
        torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.MAX, group=group)
        n_calls, has_contrast, has_anchor = int(count[0]), bool(count[1]), bool(count[2])

    mask_dtype = data["response_mask"].dtype

    def forward(ids, position, cache, keep, keep_ids=None, scored=True) -> dict:
        """One teacher forward of ``ids`` at positions ``position`` on, continuing ``cache`` (None: from
        scratch), scored at the row positions ``keep`` (on the student's ``keep_ids`` there, when given);
        an unscored forward only extends the cache, its one kept position read off the label alone."""
        input_ids = ids.unsqueeze(0)
        tensors = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "position_ids": torch.arange(position, position + ids.shape[0], device=ids.device).unsqueeze(0),
            "prompts": input_ids[:, :1],
            "responses": input_ids[:, 1:],
            "response_mask": torch.ones_like(input_ids[:, 1:], dtype=mask_dtype),
        }
        if keep_ids is not None:
            tensors["student_topk_indices"] = keep_ids.new_zeros((1, ids.shape[0] - 1, keep_ids.shape[-1]))
            tensors["student_topk_indices"][0, keep] = keep_ids
        teacher_td = left_right_2_no_padding(
            tu.get_tensordict(tensor_dict=tensors, non_tensor_dict=dict(non_tensor_dict))
        )
        keys = default_keys if scored else {**default_keys, "distillation_use_topk": False}
        tu.assign_non_tensor(teacher_td, **keys, micro_batch_size_per_gpu=1)
        teacher_td["logits_keep_positions"] = torch.nested.nested_tensor([keep], layout=torch.jagged)
        if keep_ids is not None:
            # the ids the student chose at those positions, for the head to score a chunk at a time
            teacher_td["logits_keep_topk_indices"] = torch.nested.nested_tensor([keep_ids], layout=torch.jagged)
        with engine.continuing(cache):
            return engine.infer_batch(teacher_td, loss_function=loss_function if scored else None)["model_output"]

    config = getattr(engine.module, "config", None)
    # a read keeps only its first position
    first = torch.zeros(1, dtype=torch.int64, device=rows[0][0].device)
    outputs, spans = {key: [] for key in output_keys}, []
    cache = states = None
    with engine.eval_mode(disable_auto_offload=False):
        for position, ids, branch in steps:
            if branch is None:
                if position == 0:
                    cache = TeacherCache(config=config)
                forward(ids, position, cache, first, scored=False)
                states = None
                continue
            span = branch.span
            keep = torch.arange(span.body_start, span.body_start + span.end - span.start, device=ids.device)
            keep_ids = None
            if student_topk_indices is not None:
                keep_ids = student_topk_indices[span.parent, span.start : span.end]
            if branch.point == 0:
                model_output = forward(ids, 0, None, keep, keep_ids)
            else:
                states = recurrent_states(cache) if states is None else states
                model_output = forward(ids, branch.point, cache, keep, keep_ids)
                rewind(cache, branch.point, states)
            for key in output_keys:
                outputs[key].append(model_output[key].values()[keep].float())
            spans.append(span)
        for _ in range(n_calls - len(steps)):
            forward(rows[0][0][:2], 0, None, first, scored=False)

    batch_size = data["responses"].size(0)
    response_length = max(r.shape[0] for r in data["responses"].unbind())

    def to_response_grid(key: str, contrast: bool = False, anchor: bool = False) -> torch.Tensor:
        return scatter_turn_teacher_outputs(outputs[key], spans, batch_size, response_length, contrast, anchor)

    result = {"teacher_log_probs": to_response_grid("log_probs")}
    if "all_logps" in outputs:
        result["teacher_all_log_probs"] = to_response_grid("all_logps")
    elif student_topk_indices is not None:
        result["teacher_topk_log_probs"] = to_response_grid("topk_logps")
        if has_contrast:
            result["contrast_topk_log_probs"] = to_response_grid("topk_logps", contrast=True)
        if has_anchor:
            result["anchor_topk_log_probs"] = to_response_grid("topk_logps", anchor=True)
    return result


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
        if kwargs.get("past_key_values") is not None:
            raise NotImplementedError(
                "the trust-region teacher mixes two models, which cannot continue one cache: spliced teacher rows "
                "are scored through one, so they need teacher_regularization ema"
            )
        ref_output = self.ref_module(*args, **kwargs)
        student_output = self.student_module(*args, **kwargs)
        ref_logits = self._extract_logits(ref_output)
        student_logits = self._extract_logits(student_output)
        logits = torch.lerp(ref_logits, student_logits, self.mix_coef)
        return SimpleNamespace(logits=logits)
