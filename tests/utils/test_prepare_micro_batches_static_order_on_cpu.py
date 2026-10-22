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

"""Static micro-batches (use_dynamic_bsz=False) run heaviest first, and the outputs come back in batch order."""

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.engine.utils import postprocess_batch_func, prepare_micro_batches


def _batch(seq_lens: list[int], micro_batch_size: int, force_group_size: int = 1) -> TensorDict:
    rows = [torch.full((n,), i + 1) for i, n in enumerate(seq_lens)]
    batch = TensorDict(
        {"input_ids": torch.nested.nested_tensor(rows, layout=torch.jagged)}, batch_size=[len(seq_lens)]
    )
    tu.assign_non_tensor_data(batch, "use_dynamic_bsz", False)
    tu.assign_non_tensor_data(batch, "micro_batch_size_per_gpu", micro_batch_size)
    tu.assign_non_tensor_data(batch, "force_group_size", force_group_size)
    return batch


def _row_ids(micro_batches) -> list[list[int]]:
    return [[int(row[0]) - 1 for row in mb["input_ids"].unbind()] for mb in micro_batches]


def test_rows_run_heaviest_first():
    micro_batches, idx = prepare_micro_batches(_batch([5, 30, 10, 20], micro_batch_size=1))
    assert idx == [[1], [3], [2], [0]]
    assert _row_ids(micro_batches) == idx


def test_equal_lengths_keep_batch_order():
    _, idx = prepare_micro_batches(_batch([7, 7, 7], micro_batch_size=1))
    assert idx == [[0], [1], [2]]


def test_larger_micro_batches_group_similar_rows():
    micro_batches, idx = prepare_micro_batches(_batch([5, 30, 10, 20], micro_batch_size=2))
    assert idx == [[1, 3], [2, 0]]
    assert _row_ids(micro_batches) == idx


def test_grouped_rows_keep_their_chunks():
    micro_batches, idx = prepare_micro_batches(_batch([5, 30, 10, 20], micro_batch_size=1, force_group_size=2))
    assert idx is None
    assert _row_ids(micro_batches) == [[0, 1], [2, 3]]


def test_outputs_return_in_batch_order():
    data = _batch([5, 30, 10, 20], micro_batch_size=1)
    micro_batches, idx = prepare_micro_batches(data)
    outputs = [
        {"model_output": {"lens": torch.nested.nested_tensor([mb["input_ids"].values().float()], layout=torch.jagged)}}
        for mb in micro_batches
    ]
    restored = postprocess_batch_func(outputs, idx, data)["model_output"]["lens"]
    assert [int(row[0]) for row in restored.unbind()] == [1, 2, 3, 4]
    assert [row.shape[0] for row in restored.unbind()] == [5, 30, 10, 20]
