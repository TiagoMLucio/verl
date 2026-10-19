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

import pytest
import torch

from verl.utils.grad_split import grad_split_metrics, snapshot_grads, split_sums


def _halves(rows_a, rows_b, b_noise, dim=20_000, scale=1.0, seed=0):
    """Two halves' summed gradients for rows g_i = G + noise, |G| = 1 and tr(Sigma) = b_noise."""
    gen = torch.Generator().manual_seed(seed)
    signal = torch.ones(dim, dtype=torch.float64) / dim**0.5
    rows = signal + torch.randn(rows_a + rows_b, dim, generator=gen, dtype=torch.float64) * (b_noise / dim) ** 0.5
    a, b = scale * rows[:rows_a].sum(0), scale * rows[rows_a:].sum(0)
    t = a + b
    return float(a @ a), float(t @ a), float(t @ t)


@pytest.mark.parametrize("rows_a,rows_b", [(120, 120), (100, 140)])
def test_the_noise_scale_is_recovered(rows_a, rows_b):
    metrics = grad_split_metrics(*_halves(rows_a, rows_b, b_noise=400), rows_a, rows_b)
    assert metrics["grad_split/b_noise"] == pytest.approx(400, rel=0.15)
    if rows_a == rows_b:
        assert metrics["grad_split/cos"] == pytest.approx(1 / (1 + 400 / rows_a), rel=0.15)


def test_the_loss_normalisation_cancels():
    plain = grad_split_metrics(*_halves(120, 120, b_noise=400), 120, 120)
    scaled = grad_split_metrics(*_halves(120, 120, b_noise=400, scale=1 / 240), 120, 120)
    assert scaled["grad_split/cos"] == pytest.approx(plain["grad_split/cos"])
    assert scaled["grad_split/b_noise"] == pytest.approx(plain["grad_split/b_noise"])


def test_opposed_halves_report_a_cosine_and_no_noise_scale():
    metrics = grad_split_metrics(sq_a=1.0, dot_ta=0.5, sq_t=1.0, rows_a=10, rows_b=10)
    assert metrics["grad_split/dot_ab"] == pytest.approx(-0.5) and metrics["grad_split/cos"] == pytest.approx(-0.5)
    assert "grad_split/b_noise" not in metrics


def test_split_sums_are_the_whole_model_dot_products():
    torch.manual_seed(0)
    module = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    params = list(module.parameters())
    half_a = [torch.randn_like(p) for p in params]
    half_b = [torch.randn_like(p) for p in params]
    for p, g in zip(params, half_a):
        p.grad = g.clone()
    params[-1].grad = None  # a parameter the first half never reached
    snapshot = snapshot_grads(module)
    assert len(snapshot) == len(params) - 1 and all(t.device.type == "cpu" for t in snapshot.values())
    half_a[-1] = torch.zeros_like(params[-1])
    for p, a, b in zip(params, half_a, half_b):
        p.grad = a + b
    a, t = torch.cat([g.flatten() for g in half_a]), torch.cat([(x + y).flatten() for x, y in zip(half_a, half_b)])
    sq_a, dot_ta, sq_t = split_sums(module, snapshot)
    assert (sq_a, dot_ta, sq_t) == pytest.approx((float(a @ a), float(t @ a), float(t @ t)), rel=1e-5)
