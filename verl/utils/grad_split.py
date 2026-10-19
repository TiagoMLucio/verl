# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""The split-half gradient measurement of an update: the gradient of the first half of its micro-batches (A, a
snapshot) against the rest's (B = T - A, T the accumulated gradient), read as the gradient noise scale of
McCandlish et al. 2018, "An Empirical Model of Large-Batch Training"."""

import math

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from verl.utils.device import get_device_id


def _local(grad: torch.Tensor) -> torch.Tensor:
    return grad.to_local() if isinstance(grad, DTensor) else grad


def snapshot_grads(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Each parameter's local gradient shard, copied to host memory."""
    return {
        name: _local(p.grad).detach().to("cpu", copy=True)
        for name, p in module.named_parameters()
        if p.grad is not None
    }


def split_sums(module: torch.nn.Module, half_a: dict[str, torch.Tensor], dp_group=None) -> tuple[float, float, float]:
    """|A|^2, T.A and |T|^2 over the whole model: each rank sums its own shards, then the ranks along the mesh dims
    the gradients are sharded on (``dp_group`` for plain tensors)."""
    sq_a = dot_ta = sq_t = 0.0
    groups = None
    for name, p in module.named_parameters():
        if p.grad is None:
            continue
        if groups is None:
            groups = (
                [p.grad.device_mesh.get_group(i) for i, placement in enumerate(p.grad.placements) if placement.is_shard()]
                if isinstance(p.grad, DTensor)
                else [dp_group]
            )
        t = _local(p.grad).detach().flatten().float()
        sq_t += torch.dot(t, t).item()
        a = half_a.get(name)
        if a is not None:
            a = a.to(t.device).flatten().float()
            sq_a += torch.dot(a, a).item()
            dot_ta += torch.dot(t, a).item()
    if dist.is_available() and dist.is_initialized():
        sums = torch.tensor([sq_a, dot_ta, sq_t], dtype=torch.float64, device=get_device_id())
        for group in groups or []:
            dist.all_reduce(sums, group=group)
        sq_a, dot_ta, sq_t = sums.tolist()
    return sq_a, dot_ta, sq_t


def grad_split_metrics(sq_a: float, dot_ta: float, sq_t: float, rows_a: float, rows_b: float) -> dict[str, float]:
    """With each half's gradient a mean over its rows, E[mean_A . mean_B] = |G|^2 and E|mean_A|^2 = |G|^2 +
    tr(Sigma) / rows_A; ``b_noise`` = tr(Sigma) / |G|^2 is in rows, and the loss's normalisation cancels in it and in
    ``cos``. The raw sums are kept so that steps can be pooled before the ratio is taken."""
    sq_b = sq_t - 2 * dot_ta + sq_a
    dot_ab = dot_ta - sq_a
    metrics = {
        "grad_split/sq_a": sq_a,
        "grad_split/sq_b": sq_b,
        "grad_split/dot_ab": dot_ab,
        "grad_split/rows_a": rows_a,
        "grad_split/rows_b": rows_b,
    }
    if sq_a > 0 and sq_b > 0:
        metrics["grad_split/cos"] = dot_ab / math.sqrt(sq_a * sq_b)
    if dot_ab > 0 and rows_a > 0 and rows_b > 0:
        metrics["grad_split/b_noise"] = (sq_a * rows_b + sq_b * rows_a - dot_ab * (rows_a + rows_b)) / (2 * dot_ab)
    return metrics
