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
a real row's ``teacher_seq_meta`` spans poisons ``teacher_branches`` and
``scatter_turn_teacher_outputs`` (span end far beyond the stub's response grid).
"""

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.ppo.padding_utils import construct_minimal_padding_template
from verl.trainer.ppo.sdpo.teacher_meta import DEGENERATE_META, SubRow, pack, unpack
from verl.utils import tensordict_utils as tu
from verl.workers.utils.sdpo import scatter_turn_teacher_outputs, teacher_branches


def njt(rows):
    return torch.nested.nested_tensor(rows, layout=torch.jagged)


def make_hinted_sample(resp_len=9000, prompt_len=50, contrast=False, anchor=False):
    """A spliced teacher row as the turn-hint teacher ships it: the prompt and the response once, then one
    note per sub-row, one sub-row per hinted turn (here [100, 2600) and [2600, resp_len)); with ``contrast``
    the same sub-rows again after them, a contrast text in place of the hint, and with ``anchor`` too the
    same turns once more with no note."""
    response_ids = torch.arange(resp_len, dtype=torch.long)
    prompt_ids = torch.full((prompt_len,), 7, dtype=torch.long)
    hint_ids = [torch.full((20,), 5, dtype=torch.long), torch.full((30,), 6, dtype=torch.long)]
    notes = {"hint": [], "contrast": [], "anchor": []}
    for start, hint in zip([100, 2600], hint_ids, strict=True):
        for group, text in (("hint", hint), ("contrast", hint[:10] + 3), ("anchor", hint[:0])):
            notes[group].append((start, text))
    groups = ["hint"] + ["contrast"] * contrast + ["anchor"] * anchor
    ends = {100: 2600, 2600: resp_len}
    rows = [[SubRow(prompt_len + start, text.shape[0], start, ends[start]) for start, text in notes[g]] for g in groups]
    seq = torch.cat([prompt_ids, response_ids, *(text for g in groups for _, text in notes[g])])
    return {
        "responses": response_ids,
        "response_mask": torch.ones(resp_len),
        "teacher_input_ids": seq,
        "teacher_seq_meta": torch.tensor(pack(prompt_len, *rows), dtype=torch.int64),
    }


STUB = {
    "responses": torch.tensor([11], dtype=torch.long),
    "response_mask": torch.zeros(1),
    "teacher_input_ids": torch.tensor([8, 11], dtype=torch.long),
    "teacher_seq_meta": torch.tensor(DEGENERATE_META, dtype=torch.int64),
}


def branch_spans(micro):
    return [branch.span for _, branches in teacher_branches(
        micro["teacher_input_ids"], micro["teacher_seq_meta"], micro["responses"]
    ) for branch in branches]


def run_teacher_grid_path(micro, value=lambda span: torch.randn(())):
    """Mirror _score_teacher_branches's shape flow: every branch's outputs cover its own span."""
    batch_size = micro["responses"].size(0)
    full_response_length = max(r.shape[0] for r in micro["responses"].unbind())
    spans = branch_spans(micro)
    outputs = [value(span).expand(span.end - span.start).clone() for span in spans]
    grids = [
        scatter_turn_teacher_outputs(outputs, spans, batch_size, full_response_length, contrast, anchor)
        for contrast, anchor in ((False, False), (True, False), (False, True))
    ]
    return spans, grids


def to_td(samples):
    keys = ("responses", "response_mask", "teacher_input_ids", "teacher_seq_meta")
    return TensorDict({k: njt([s[k].float() if k == "response_mask" else s[k] for s in samples]) for k in keys},
                      batch_size=len(samples))


def test_micro_splits_stay_row_aligned():
    td = to_td([make_hinted_sample(), STUB])
    run_teacher_grid_path(td)
    for idx in ([0], [1], [1, 0]):
        run_teacher_grid_path(tu.index_select_tensor_dict(td, idx))
    for micro in tu.chunk_tensordict(td, 2):
        run_teacher_grid_path(micro)


# each group's sub-row of each turn, and the stub, scored with its own constant
VALUES = {
    (0, 100, False, False): 1.0, (0, 2600, False, False): 2.0, (0, 100, True, False): 3.0,
    (0, 2600, True, False): 4.0, (0, 100, False, True): 5.0, (0, 2600, False, True): 6.0, (1, 0, False, False): 7.0,
}


def _value(span):
    return torch.tensor(VALUES[(span.parent, span.start, span.contrast, span.anchor)])


def test_contrast_subrows_scatter_apart_from_the_hint_ones():
    spans, (hint, contrast, _) = run_teacher_grid_path(to_td([make_hinted_sample(contrast=True), STUB]), _value)
    assert sorted((span.parent, span.contrast) for span in spans) == [
        (0, False), (0, False), (0, True), (0, True), (1, False)
    ]
    assert hint[0, 100:2600].eq(1).all() and hint[0, 2600:].eq(2).all() and hint[0, :100].eq(0).all()
    assert contrast[0, 100:2600].eq(3).all() and contrast[0, 2600:].eq(4).all() and contrast[0, :100].eq(0).all()
    assert hint[1, 0] == 7 and contrast[1].eq(0).all(), "the un-hinted row's stub has no contrast sub-row"


def test_anchor_subrows_scatter_apart_from_the_hint_and_contrast_ones():
    spans, (hint, contrast, anchor) = run_teacher_grid_path(
        to_td([make_hinted_sample(contrast=True, anchor=True), STUB]), _value
    )
    assert sorted((span.parent, span.contrast, span.anchor) for span in spans) == [
        (0, False, False), (0, False, False), (0, False, True), (0, False, True),
        (0, True, False), (0, True, False), (1, False, False),
    ]
    assert hint[0, 100:2600].eq(1).all() and hint[0, 2600:].eq(2).all()
    assert contrast[0, 100:2600].eq(3).all() and contrast[0, 2600:].eq(4).all()
    assert anchor[0, 100:2600].eq(5).all() and anchor[0, 2600:].eq(6).all() and anchor[0, :100].eq(0).all()
    assert hint[1, 0] == 7 and anchor[1].eq(0).all(), "the un-hinted row's stub has no anchor sub-row"


def test_the_meta_packs_and_unpacks_three_groups():
    rows = [SubRow(8, 2, 2, 8), SubRow(14, 3, 10, 16)]
    contrast = [SubRow(8, 4, 2, 8), SubRow(14, 5, 10, 16)]
    anchor = [SubRow(8, 0, 2, 8), SubRow(14, 0, 10, 16)]
    flat = pack(6, rows, contrast, anchor)
    assert flat[:2] == [2, 6] and len(flat) == 2 + 3 * 2 * 4
    assert unpack(flat) == rows and unpack(flat, contrast=True) == contrast and unpack(flat, anchor=True) == anchor
    two = pack(6, rows, contrast)
    assert unpack(two, contrast=True) == contrast and unpack(two, anchor=True) == []
    with pytest.raises(AssertionError, match="anchor"):
        pack(6, rows, (), anchor)
    with pytest.raises(AssertionError, match="one group"):
        unpack(flat, contrast=True, anchor=True)


def test_corrupt_spans_are_caught():
    sample = make_hinted_sample()
    sample["responses"] = torch.arange(100, dtype=torch.long)  # truncated vs meta
    sample["response_mask"] = torch.ones(100)
    with pytest.raises(AssertionError, match="teacher_seq_meta malformed"):
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

    assert template["teacher_seq_meta"].tolist() == DEGENERATE_META
    assert template["teacher_input_ids"].shape[0] == 2
    assert template["self_distillation_mask"].tolist() == [0.0]
    assert tag["is_padding"]

    # one real hinted row + one padding row survives the grid path whole and under every rank split
    mixed = to_td([source, template])
    run_teacher_grid_path(mixed)
    for idx in ([0], [1]):
        run_teacher_grid_path(tu.index_select_tensor_dict(mixed, idx))


def test_a_pad_made_after_the_advantage_pass_sits_on_its_own_grid():
    """The update re-pads after dropping unsupervised rows, when the source already carries the
    log-prob and advantage passes' fields; a pad inheriting them has a 1-token response with
    9000-token advantages, which the SDPO row cut then cannot align with the student."""
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
        old_log_probs=torch.randn(9000),
        entropy=torch.rand(9000),
        advantages=torch.randn(9000),
        returns=torch.randn(9000),
        uid="real-0",
        num_turns=40,
    )
    template, _ = construct_minimal_padding_template(source, {"seq_len": 9050}, eos_token_id=2)

    for key in ("old_log_probs", "entropy", "advantages", "returns", "response_mask", "loss_mask"):
        assert template[key].shape == (1,), key
        assert not template[key].any(), key
    assert template["responses"].tolist() == [2] and template["prompts"].tolist() == [2]
    assert template["input_ids"].tolist() == [2, 2]
    assert template["teacher_seq_meta"].tolist() == list(DEGENERATE_META)
