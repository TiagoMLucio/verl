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
"""traj-mean-token-mean under condensation: the update batch the trainer hands the worker,
through ``_balance_batch`` -> ``_drop_unsupervised_rows`` -> ``_update_actor`` over an in-memory
TransferQueue, at the production shape (batch 128, mini 128, n 1, dp 4)."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.trainer import main_ppo_sync
from verl.trainer.ppo import padding_utils

DP, MINI = 4, 128


@dataclass
class KVBatchMeta:
    keys: list
    tags: list
    partition_id: str
    fields: object = None
    extra_info: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.keys)

    def reorder(self, indices):
        self.keys = [self.keys[i] for i in indices]
        self.tags = [self.tags[i] for i in indices]


class KV:
    """Per-key sample dicts; get stacks tensor fields (jagged when lengths differ)."""

    def __init__(self):
        self.store = {}

    def kv_batch_get(self, keys, partition_id, select_fields=None):
        rows = [self.store[k] for k in keys]
        out = {}
        for f in select_fields or list(rows[0].keys()):
            vals = [r[f] for r in rows]
            if isinstance(vals[0], torch.Tensor):
                try:
                    out[f] = torch.stack(vals)
                except RuntimeError:
                    out[f] = torch.nested.nested_tensor(vals, layout=torch.jagged)
            else:
                out[f] = vals
        return TensorDict(out, batch_size=len(keys))

    def kv_batch_put(self, keys, partition_id, fields, tags=None):
        for i, k in enumerate(keys):
            self.store.setdefault(k, {})
            for f in fields.keys():
                v = fields[f]
                self.store[k][f] = v[i] if isinstance(v, torch.Tensor | list) else v

    def kv_clear(self, keys, partition_id):
        for k in keys:
            self.store.pop(k, None)


def _row(weight, traj, seq_len):
    return dict(
        prompts=torch.tensor([1, 2]),
        responses=torch.arange(3, 3 + seq_len - 2),
        input_ids=torch.arange(1, 1 + seq_len),
        attention_mask=torch.ones(seq_len, dtype=torch.int64),
        position_ids=torch.arange(seq_len),
        response_mask=torch.ones(seq_len - 2, dtype=torch.int64),
        loss_mask=torch.ones(seq_len - 2, dtype=torch.int64),
        rm_scores=torch.zeros(seq_len - 2),
        rollout_log_probs=torch.zeros(seq_len - 2),
        num_turns=1,
        trace_weight=torch.tensor([weight]),
        traj_id=torch.tensor([traj]),
        teacher_input_ids=torch.tensor([1, 2]),
        teacher_seq_meta=torch.tensor([1, 2, 1, 0, 0, 1]),
        self_distillation_mask=torch.full((seq_len - 2,), weight),
        uid=f"u{traj}",
    )


def _trainer(monkeypatch, n_tasks, n_rows, supervised_rows, micro=None):
    """``n_rows - n_tasks`` tasks condensed into two segments; the first ``supervised_rows``
    rows carry weight. Returns the trainer, the balanced batch, the captured update calls and the KV."""
    monkeypatch.setattr(main_ppo_sync, "KVBatchMeta", KVBatchMeta)
    monkeypatch.setattr(padding_utils, "KVBatchMeta", KVBatchMeta)
    kv = KV()
    monkeypatch.setattr(main_ppo_sync, "tq", kv)
    monkeypatch.setattr(padding_utils, "tq", kv)

    keys, tags, traj = [], [], []
    condensed = n_rows - n_tasks
    for t in range(n_tasks):
        for s in range(2 if t < condensed else 1):
            keys.append(f"u{t}_0_{s}")
            tags.append({"seq_len": 8 + (t % 5)})
            traj.append(t)
    for i, k in enumerate(keys):
        kv.store[k] = _row(1.0 if i < supervised_rows else 0.0, traj[i], tags[i]["seq_len"])

    trainer = object.__new__(main_ppo_sync.PPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "ppo_mini_batch_size": MINI,
                    "ppo_micro_batch_size_per_gpu": micro,
                    "ppo_epochs": 1,
                    "drop_unsupervised_rows": True,
                    "loss_agg_mode": "traj-mean-token-mean",
                    "policy_loss": {"loss_mode": "sdpo"},
                    "self_distillation": {"full_logit_distillation": False},
                    "calculate_entropy": False,
                    "entropy_coeff": 0.0,
                    "data_loader_seed": 1,
                    "shuffle": True,
                },
                "rollout": {"n": 1, "temperature": 1.0, "log_prob_micro_batch_size_per_gpu": micro},
            },
            "trainer": {"critic_warmup": 0},
        }
    )
    trainer.use_critic = False
    trainer.global_steps = 1
    trainer.tokenizer = SimpleNamespace(eos_token_id=0)
    trainer._get_dp_size = lambda wg, role: DP
    removed = []
    trainer.replay_buffer = SimpleNamespace(removed=removed, remove=lambda partition_id, keys: removed.extend(keys))
    sent = []
    trainer.actor_rollout_wg = SimpleNamespace(update_actor=lambda b: sent.append(b) or {"metrics": {"mfu": 0.0}})
    batch = trainer._balance_batch(KVBatchMeta(keys=keys, tags=tags, partition_id="train"), metrics={})
    return trainer, batch, sent, kv


CASES = [
    # (tasks, rows, supervised rows) -> (rows after balance, mini-batch rows sent, supervised trajectories);
    # one mini-batch per step, so the padding only rounds each up to a multiple of dp
    pytest.param(128, 128, 90, 128, 92, 90, id="no condensation, 90 hinted"),
    pytest.param(128, 150, 100, 152, 100, 78, id="22 condensed, 100 hinted rows"),
    pytest.param(128, 150, 130, 152, 132, 108, id="22 condensed, 130 hinted rows"),
    pytest.param(128, 129, 129, 132, 132, 128, id="1 condensed, all 129 hinted"),
]


@pytest.mark.parametrize("n_tasks, n_rows, supervised_rows, balanced, mini_rows, trajectories", CASES)
def test_traj_mean_update_batch_is_one_mini_batch(
    monkeypatch, n_tasks, n_rows, supervised_rows, balanced, mini_rows, trajectories
):
    trainer, batch, sent, kv = _trainer(monkeypatch, n_tasks, n_rows, supervised_rows)
    assert len(batch) == balanced
    metrics = {}
    assert trainer._update_actor(batch, metrics) is batch
    (update_batch,) = sent
    assert len(update_batch) == mini_rows
    assert update_batch.extra_info["mini_batch_size"] == mini_rows, "one mini-batch per rank"
    assert update_batch.extra_info["mini_batch_size"] % DP == 0
    assert update_batch.extra_info["global_batch_size"] == trajectories
    # real rows only: the balance's own padding is not a dropped row
    assert metrics["self_distillation/dropped_unsupervised_rows"] == n_rows - supervised_rows
    assert "actor/skipped_update" not in metrics
    # the drop's re-pad rows exist only in the update batch, and fit clears only the batch's keys
    stray = [k for k in update_batch.keys if k not in set(batch.keys)]
    assert len(stray) == mini_rows - supervised_rows
    assert not any(k in kv.store for k in stray)
    assert trainer.replay_buffer.removed == stray


@pytest.mark.parametrize(
    "n_tasks, n_rows",
    [
        pytest.param(128, 128, id="no condensation, nothing hinted"),
        pytest.param(128, 150, id="22 condensed, nothing hinted"),
    ],
)
def test_traj_mean_skips_the_update_without_supervision(monkeypatch, n_tasks, n_rows):
    trainer, batch, sent, _ = _trainer(monkeypatch, n_tasks, n_rows, supervised_rows=0)
    metrics = {}
    assert trainer._update_actor(batch, metrics) is batch
    assert sent == []
    assert metrics["actor/skipped_update"] == 1.0


def test_dropped_batch_is_length_balanced_across_dp(monkeypatch):
    """After the drop the padding rows are spread by _balance_batch, not stacked on the last rank."""
    trainer, batch, sent, _ = _trainer(monkeypatch, 128, 150, 97)
    metrics = {}
    trainer._update_actor(batch, metrics)
    (update_batch,) = sent
    per_rank = len(update_batch) // DP
    padding_per_rank = [
        sum(bool(t.get("is_padding", False)) for t in update_batch.tags[r * per_rank : (r + 1) * per_rank])
        for r in range(DP)
    ]
    assert sum(padding_per_rank) == 3
    assert max(padding_per_rank) == 1
    assert metrics["update_seqlen/balanced_max"] - metrics["update_seqlen/balanced_min"] <= 4


def test_the_multiple_follows_the_micro_batch_size(monkeypatch):
    """Each rank's share has to split into whole micro-batches: dp x micro rows, not the mini-batch."""
    trainer, batch, sent, _ = _trainer(monkeypatch, 128, 150, 100, micro=2)
    assert len(batch) == 152
    trainer._update_actor(batch, {})
    (update_batch,) = sent
    assert len(update_batch) == 104
    assert (len(update_batch) // DP) % 2 == 0
