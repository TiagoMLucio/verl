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
"""SDPO turn-mode consistency across micro splitting and batch padding.

Covers the dp>1 crash where ``upsample_batch_to_divisible_size`` padding rows
inherited the template sample's spliced teacher fields: a 1-token stub carrying
a real row's ``teacher_seq_meta`` spans poisons ``scatter_turn_teacher_outputs``
(span end far beyond the stub's response grid).
"""

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.ppo import sdpo_teacher
from verl.trainer.ppo.padding_utils import construct_minimal_padding_template
from verl.utils import tensordict_utils as tu
from verl.workers.utils.sdpo import explode_turn_teacher_rows, scatter_turn_teacher_outputs


def njt(rows):
    return torch.nested.nested_tensor(rows, layout=torch.jagged)


def make_hinted_sample(resp_len=9000, prompt_len=50):
    response_ids = torch.arange(resp_len, dtype=torch.long)
    prompt_ids = torch.full((prompt_len,), 7, dtype=torch.long)
    hinted = [(3, 100, 2600, "h1"), (7, 2600, resp_len, "h2")]
    hint_ids = [torch.full((20,), 5, dtype=torch.long), torch.full((30,), 6, dtype=torch.long)]
    header_ids = torch.tensor([90, 91, 92], dtype=torch.long)
    seq, meta, _ = sdpo_teacher.build_spliced_teacher_row(
        prompt_ids, response_ids, hinted, hint_ids, 32768, header_ids
    )
    return {
        "responses": response_ids,
        "response_mask": torch.ones(resp_len),
        "teacher_input_ids": seq,
        "teacher_seq_meta": torch.tensor(meta, dtype=torch.int64),
    }


def run_teacher_grid_path(micro):
    """Mirror _compute_sdpo_teacher_logps_for_loss's turn-mode shape flow."""
    batch_size = micro["responses"].size(0)
    full_response_length = max(r.shape[0] for r in micro["responses"].unbind())
    _, sub_resps, _, parents, spans = explode_turn_teacher_rows(
        teacher_input_ids=micro["teacher_input_ids"],
        teacher_seq_meta=micro["teacher_seq_meta"],
        responses=micro["responses"],
        response_mask=micro["response_mask"],
    )
    if not parents:
        # mirror the worker's hints-only guard: no teacher forward, zero grid
        return torch.zeros(batch_size, full_response_length)
    max_body = max(r.shape[0] for r in sub_resps.unbind())
    fake_outputs = torch.randn(len(parents), max_body)
    return scatter_turn_teacher_outputs(fake_outputs, parents, spans, batch_size, full_response_length)


def to_td(samples):
    keys = ("responses", "response_mask", "teacher_input_ids", "teacher_seq_meta")
    return TensorDict({k: njt([s[k].float() if k == "response_mask" else s[k] for s in samples]) for k in keys},
                      batch_size=len(samples))


def test_micro_splits_stay_row_aligned():
    stub = {
        "responses": torch.tensor([11], dtype=torch.long),
        "response_mask": torch.zeros(1),
        "teacher_input_ids": torch.tensor([8, 11], dtype=torch.long),
        "teacher_seq_meta": torch.tensor([1], dtype=torch.int64),
    }
    td = to_td([make_hinted_sample(), stub])
    run_teacher_grid_path(td)
    for idx in ([0], [1], [1, 0]):
        run_teacher_grid_path(tu.index_select_tensor_dict(td, idx))
    for micro in tu.chunk_tensordict(td, 2):
        run_teacher_grid_path(micro)


def test_corrupt_spans_are_caught():
    sample = make_hinted_sample()
    sample["responses"] = torch.arange(100, dtype=torch.long)  # truncated vs meta
    sample["response_mask"] = torch.ones(100)
    with pytest.raises(AssertionError, match="spans exceed the response row"):
        run_teacher_grid_path(to_td([sample]))


def test_padding_template_rebuilds_teacher_fields():
    source = make_hinted_sample()
    source.update(
        prompts=torch.full((50,), 7, dtype=torch.long),
        input_ids=torch.arange(9050, dtype=torch.long),
        attention_mask=torch.ones(9050, dtype=torch.long),
        position_ids=torch.arange(9050),
        loss_mask=torch.ones(9000),
        self_distillation_mask=torch.ones(9000),
        rm_scores=torch.zeros(9000),
        rollout_log_probs=torch.zeros(9000),
        uid="real-0",
        num_turns=40,
    )
    template, tag = construct_minimal_padding_template(source, {"seq_len": 9050}, eos_token_id=2)

    assert template["teacher_seq_meta"].tolist() == [1]
    assert template["teacher_input_ids"].shape[0] == 2
    assert template["self_distillation_mask"].tolist() == [0.0]
    assert tag["is_padding"]

    # one real hinted row + one padding row survives the grid path whole and under every rank split
    mixed = to_td([source, template])
    run_teacher_grid_path(mixed)
    for idx in ([0], [1]):
        run_teacher_grid_path(tu.index_select_tensor_dict(mixed, idx))
