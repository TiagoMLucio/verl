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
"""A validation session that never comes back counts as a failed sample: accuracy stays over every
dispatched session (prompts x val n), not over the ones that returned."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer import main_ppo_sync


class KV:
    def __init__(self):
        self.store = {}

    def kv_batch_get(self, keys, partition_id, select_fields):
        rows = [self.store[k] for k in keys]
        out = {}
        for f in select_fields:
            vals = [r[f] for r in rows]
            if f in ("prompts", "responses"):
                out[f] = torch.nested.nested_tensor(vals, layout=torch.jagged)
            elif isinstance(vals[0], torch.Tensor):
                out[f] = torch.stack(vals)
            else:
                out[f] = np.array(vals, dtype=object)
        return out

    def kv_clear(self, keys, partition_id):
        for k in keys:
            self.store.pop(k, None)


def _validate(monkeypatch, returned_sessions, n_prompts=2, val_n=2):
    """Each prompt's sessions in ``returned_sessions[prompt]`` come back solved; the rest never return."""
    kv = KV()
    monkeypatch.setattr(main_ppo_sync, "tq", kv)
    monkeypatch.setattr(
        main_ppo_sync, "tu", SimpleNamespace(get_tensordict=lambda d: d, assign_non_tensor_data=lambda *a: None)
    )
    trainer = object.__new__(main_ppo_sync.PPOTrainer)
    trainer.global_steps = 4
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"val_kwargs": {"n": val_n}}},
            "trainer": {"log_val_generations": 0, "validation_data_dir": None},
        }
    )
    trainer.tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda ids, skip_special_tokens: "text")
    sampled = []

    def generate_sequences(batch):
        for p, uid in enumerate(batch["uid"]):
            for s in returned_sessions[p]:
                key = f"{uid}_{s}_0"
                sampled.append(key)
                kv.store[key] = {
                    "prompts": torch.tensor([1, 2]),
                    "responses": torch.tensor([3, 4, 5]),
                    "uid": uid,
                    "rm_scores": torch.tensor([0.0, 0.0, 1.0]),
                    "num_turns": 3,
                    "reward_model": {"ground_truth": f"g{p}"},
                    "data_source": "v128",
                    "extra_fields": {"traj_exit_reason": "finished", "reward_extra_info": {"acc": 1.0}},
                }

    trainer.async_rollout_manager = SimpleNamespace(generate_sequences=generate_sequences)
    trainer.replay_buffer = SimpleNamespace(
        sample=lambda partition_id, global_steps: SimpleNamespace(keys=list(sampled), partition_id=partition_id),
        remove=lambda partition_id, keys: sampled.clear(),
    )
    trainer.val_dataloader = [
        {
            "raw_prompt": np.array([f"p{i}" for i in range(n_prompts)], dtype=object),
            "data_source": np.array(["v128"] * n_prompts, dtype=object),
            "reward_model": np.array([{"ground_truth": f"g{i}"} for i in range(n_prompts)], dtype=object),
        }
    ]
    return trainer._validate()


def test_a_missing_session_counts_as_a_failed_sample(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger=main_ppo_sync.logger.name):
        out = _validate(monkeypatch, returned_sessions=[[0, 1], [0]])
    # 3 of 4 sessions solved: the missing one pulls both reward and acc down to 3/4, not 3/3
    assert out["val-core/v128/acc/mean@2"] == pytest.approx(0.75)
    assert out["val-aux/v128/reward/mean@2"] == pytest.approx(0.75)
    assert out["val-aux/exit_missing_fraction"] == pytest.approx(0.25)
    assert out["val-aux/exit_finished_fraction"] == pytest.approx(0.75)
    assert out["val-aux/num_turns/min"] == 0
    assert "1 of 4 dispatched sessions never returned" in caplog.text


def test_the_missing_fraction_is_emitted_at_zero(monkeypatch):
    out = _validate(monkeypatch, returned_sessions=[[0, 1], [0, 1]])
    assert out["val-aux/exit_missing_fraction"] == 0.0
    assert out["val-core/v128/acc/mean@2"] == 1.0


def test_a_batch_where_nothing_returned_scores_zero_instead_of_crashing(monkeypatch):
    out = _validate(monkeypatch, returned_sessions=[[], []])
    assert out["val-core/v128/reward/mean@2"] == 0.0
    assert out["val-aux/exit_missing_fraction"] == 1.0
