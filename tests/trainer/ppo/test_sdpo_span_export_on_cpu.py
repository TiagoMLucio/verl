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
"""The per-span export: the aggregates each scored span contributes to the batch metrics, and
the trainer step that carries them from the update into the rollout dump."""

import json
from types import SimpleNamespace

import pytest
import torch

from verl.trainer import main_ppo_sync
from verl.trainer.ppo.core_algos import SDPO_SPAN_ROWS_KEY, _distillation_span_rows, compute_self_distillation_loss

# position 0 and 5 are outside every span or unmasked, and carry values that would show up
PER_TOKEN_LOSS = torch.tensor([[10.0, 1.0, 2.0, 3.0, 4.0, 100.0], [7.0, 7.0, 7.0, 7.0, 7.0, 7.0]])
LOSS_MASK = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
STUDENT = torch.zeros(2, 6)
TEACHER = torch.tensor([[9.0, -1.0, 2.0, -3.0, 4.0, 9.0], [1.0, 3.0, 9.0, 9.0, 9.0, 9.0]])
# argmax disagreement on the odd positions of row 0, on none of row 1
STUDENT_TOPK = torch.tensor([[[1.0, 0.0]] * 6, [[1.0, 0.0]] * 6])
TEACHER_TOPK = torch.tensor(
    [
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0]] * 6,
    ]
)


def span_rows(spans, with_topk=True):
    return _distillation_span_rows(
        per_token_loss=PER_TOKEN_LOSS,
        loss_mask=LOSS_MASK,
        student_log_probs=STUDENT,
        teacher_log_probs=TEACHER,
        spans=spans,
        student_topk_log_probs=STUDENT_TOPK if with_topk else None,
        teacher_topk_log_probs=TEACHER_TOPK if with_topk else None,
    )


def test_span_row_aggregates_only_its_own_masked_positions():
    """Row 0's span covers |gap| 1, 2, 3, 4 and losses 1, 2, 3, 4; the poisoned positions
    outside it (and the masked ones inside) must not reach any of the four numbers."""
    (row,) = span_rows([(7, 0, 1, 5)])
    assert row == {
        "row_id": 7,
        "start": 1,
        "end": 5,
        "supervised_tokens": 4,
        "absgap_mean": 2.5,
        "loss_p50": 2.5,
        "teacher_prefers_other_token": 0.5,
    }


def test_spans_are_reported_per_span_not_per_row():
    rows = span_rows([(3, 0, 1, 3), (3, 0, 3, 5), (4, 1, 0, 2)])
    assert [(r["row_id"], r["start"], r["end"], r["supervised_tokens"]) for r in rows] == [
        (3, 1, 3, 2),
        (3, 3, 5, 2),
        (4, 0, 2, 2),
    ]
    # |gap| 1, 2 then 3, 4 on row 0; 1, 3 on row 1, and only row 0's argmax ever disagrees
    assert [r["absgap_mean"] for r in rows] == [1.5, 3.5, 2.0]
    assert [r["loss_p50"] for r in rows] == [1.5, 3.5, 7.0]
    assert [r["teacher_prefers_other_token"] for r in rows] == [0.5, 0.5, 0.0]


def test_unsupervised_spans_are_dropped_and_topk_is_optional():
    # the un-hinted row's stub span: inside the response grid, but no supervised token
    assert span_rows([(1, 1, 2, 3)]) == []
    (row,) = span_rows([(1, 1, 0, 2)], with_topk=False)
    assert "teacher_prefers_other_token" not in row


def _loss_metrics(supervised_spans):
    cfg = SimpleNamespace(
        full_logit_distillation=False, distillation_topk=None, distillation_add_tail=False, alpha=1.0, is_clip=None
    )
    _, metrics = compute_self_distillation_loss(
        student_log_probs=STUDENT,
        teacher_log_probs=TEACHER,
        response_mask=torch.ones(2, 6),
        self_distillation_config=cfg,
        self_distillation_mask=LOSS_MASK,
        supervised_spans=supervised_spans,
    )
    return metrics


def test_the_loss_exports_spans_only_when_it_is_given_them():
    assert SDPO_SPAN_ROWS_KEY not in _loss_metrics(None)
    assert SDPO_SPAN_ROWS_KEY not in _loss_metrics([(0, 1, 2, 6)])
    (row,) = _loss_metrics([(9, 0, 1, 5)])[SDPO_SPAN_ROWS_KEY]
    assert (row["row_id"], row["supervised_tokens"], row["absgap_mean"]) == (9, 4, 2.5)


class RowIdStub:
    def __init__(self, row_ids=None):
        self.row_ids = row_ids

    def kv_batch_get(self, keys, partition_id, select_fields):
        if self.row_ids is None:
            raise KeyError(select_fields[0])
        return {"row_id": torch.tensor(self.row_ids, dtype=torch.int64).unsqueeze(-1)}


def _trainer(span_rows_out):
    trainer = object.__new__(main_ppo_sync.PPOTrainer)
    trainer.global_steps = 7
    trainer._sdpo_span_rows = span_rows_out
    return trainer


SPAN_A = {"row_id": 2, "start": 0, "end": 4, "supervised_tokens": 4, "absgap_mean": 0.5, "loss_p50": 0.25}
SPAN_B = {"row_id": 0, "start": 3, "end": 9, "supervised_tokens": 6, "absgap_mean": 1.5, "loss_p50": 0.75}


# the three dumped rows carry these row ids, so SPAN_A belongs to key "b" and SPAN_B to key "a"
ROW_IDS = [0, 2, 5]


def test_span_column_lands_each_span_on_the_row_it_came_from(monkeypatch):
    trainer = _trainer([SPAN_A, SPAN_B])
    monkeypatch.setattr(main_ppo_sync, "tq", RowIdStub(ROW_IDS))
    batch = SimpleNamespace(keys=["a", "b", "c"], partition_id="train")

    # the dump writes rows sorted by key, so the column follows sorted_indices, not row order
    column = [json.loads(entry) for entry in trainer._supervised_span_column(batch, [2, 0, 1], [100, 101, 102])]

    assert [len(spans) for spans in column] == [0, 1, 1]
    assert column[1][0] == {
        "start": 3, "end": 9, "supervised_tokens": 6, "absgap_mean": 1.5, "loss_p50": 0.75,
        "step": 7, "sample_index": 100,
    }
    assert column[2][0]["sample_index"] == 101 and column[2][0]["start"] == 0
    assert SPAN_A["row_id"] == 2, "the exported rows are copied, not consumed"


def test_span_column_without_sample_indices_still_carries_the_step(monkeypatch):
    trainer = _trainer([SPAN_A])
    monkeypatch.setattr(main_ppo_sync, "tq", RowIdStub(ROW_IDS))
    batch = SimpleNamespace(keys=["a", "b", "c"], partition_id="train")

    column = [json.loads(entry) for entry in trainer._supervised_span_column(batch, [0, 1, 2], None)]

    assert column[1] == [dict(start=0, end=4, supervised_tokens=4, absgap_mean=0.5, loss_p50=0.25, step=7)]
    assert column[0] == [] and column[2] == []


@pytest.mark.parametrize("rows, row_ids", [([], ROW_IDS), ([SPAN_A], None)])
def test_span_column_is_absent_without_spans_or_row_ids(monkeypatch, rows, row_ids):
    trainer = _trainer(rows)
    monkeypatch.setattr(main_ppo_sync, "tq", RowIdStub(row_ids))
    batch = SimpleNamespace(keys=["a", "b", "c"], partition_id="train")

    assert trainer._supervised_span_column(batch, [0, 1, 2], [100, 101, 102]) is None
