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
"""Refill: a GRPO group whose sessions all share one score is set aside and replaced by new prompts while
the step generates, so the update gets a batch of groups that carry an advantage."""

import threading
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer import main_ppo_sync

MIXED, FAIL, PASS = [1.0, 0.0, 0.0, 1.0], [0.0] * 4, [1.0] * 4


class Step:
    """One step's prompt groups in a buffer without its poll thread: each group finishes on a given poll."""

    def __init__(self, monkeypatch, refills=(), delay=1, step=7):
        monkeypatch.setattr(main_ppo_sync, "KVBatchMeta", SimpleNamespace)
        monkeypatch.setattr(main_ppo_sync, "time", SimpleNamespace(sleep=self._poll, monotonic=time.monotonic))
        monkeypatch.setattr(main_ppo_sync, "tq", SimpleNamespace(kv_batch_get=self._scores))
        self.buffer = object.__new__(main_ppo_sync.ReplayBuffer)
        self.buffer.partitions = defaultdict(dict)
        self.buffer.owners = defaultdict(dict)
        self.buffer.poll_interval = 0.0
        self.buffer.liveness_interval = 1e9
        self.buffer.lock = threading.Lock()
        self.step, self.delay, self.polls = step, delay, 0
        self.refills, self.calls = list(refills), []
        self.schedule, self.score = defaultdict(list), {}

    def dispatch(self, uid, scores, finish_at):
        self.buffer.add("train", {uid: {"global_steps": self.step, "status": "running"}})
        self.schedule[finish_at].append((uid, scores))

    def refill(self, count):
        self.calls.append(count)
        sent = 0
        while sent < count and self.refills:
            self.dispatch(f"r{len(self.calls)}.{sent}", self.refills.pop(0), self.polls + self.delay)
            sent += 1
        return sent

    def sample(self, num_groups, max_prompts=0):
        return self.buffer.sample_varied_groups("train", self.step, num_groups, self.refill, max_prompts)

    def _poll(self, _):
        self.polls += 1
        for uid, scores in self.schedule.pop(self.polls, []):
            rows = {f"{uid}_{s}_0": {"global_steps": self.step, "status": "success"} for s in range(len(scores))}
            self.score.update({f"{uid}_{s}_0": score for s, score in enumerate(scores)})
            self.buffer.add("train", {**rows, uid: {"global_steps": self.step, "status": "finished"}})

    def _scores(self, keys, partition_id, select_fields):
        return {"rm_scores": torch.tensor([[self.score[k]] for k in keys])}


def _uids(batch):
    return list(dict.fromkeys(main_ppo_sync._parse_key(k)[0] for k in batch.keys))


def test_groups_without_signal_are_replaced_and_the_varied_ones_kept_in_dispatch_order(monkeypatch):
    step = Step(monkeypatch, refills=[MIXED, FAIL, MIXED, MIXED, FAIL, PASS])
    for uid, scores, at in (("a", MIXED, 1), ("b", FAIL, 1), ("c", MIXED, 2), ("d", PASS, 2)):
        step.dispatch(uid, scores, at)
    kept, dropped, metrics = step.sample(num_groups=4)
    # b earns 2 refills, but only 2 slots are free while c and d run; then d and r1.1 earn 4
    assert step.calls == [2, 4]
    assert _uids(kept) == ["a", "c", "r1.0", "r2.0"]
    assert _uids(dropped) == ["b", "d", "r1.1", "r2.1", "r2.2", "r2.3"]
    assert len(kept.keys) == 16 and len(dropped.keys) == 24
    assert metrics["training/refill/dispatched_prompts"] == 10
    assert metrics["training/refill/varied_groups"] == 5
    assert metrics["training/refill/surplus_varied_groups"] == 1
    assert metrics["training/refill/all_pass_groups"] == 2
    assert metrics["training/refill/all_fail_groups"] == 3
    assert metrics["training/refill/no_signal_groups_kept"] == 0
    assert metrics["training/refill/score_mean"] == pytest.approx(np.mean(MIXED * 5 + FAIL * 3 + PASS * 2))


def test_enough_varied_groups_stop_the_refills_and_the_running_ones_are_awaited(monkeypatch):
    step = Step(monkeypatch, refills=[MIXED, MIXED, MIXED])
    step.dispatch("a", FAIL, 1)
    step.dispatch("b", MIXED, 5)
    kept, dropped, metrics = step.sample(num_groups=2)
    # one slot per poll while b runs; r1 and r2 make two varied groups, so nothing more is sent
    assert step.calls == [1, 1]
    assert step.polls == 5
    assert _uids(kept) == ["b", "r1.0"]
    assert _uids(dropped) == ["a", "r2.0"]
    assert metrics["training/refill/surplus_varied_groups"] == 1


def test_an_empty_prompt_source_fills_the_batch_with_groups_without_signal(monkeypatch):
    step = Step(monkeypatch)
    step.dispatch("a", MIXED, 1)
    step.dispatch("b", PASS, 1)
    kept, dropped, metrics = step.sample(num_groups=2)
    assert step.calls == [2]
    assert _uids(kept) == ["a", "b"] and dropped.keys == []
    assert metrics["training/refill/no_signal_groups_kept"] == 1


def test_max_prompts_caps_what_one_step_sends(monkeypatch):
    step = Step(monkeypatch, refills=[FAIL, MIXED, MIXED])
    step.dispatch("a", FAIL, 1)
    step.dispatch("b", FAIL, 1)
    kept, dropped, metrics = step.sample(num_groups=2, max_prompts=3)
    assert step.calls == [1]
    assert metrics["training/refill/dispatched_prompts"] == 3
    assert _uids(kept) == ["a", "b"] and _uids(dropped) == ["r1.0"]
    assert metrics["training/refill/no_signal_groups_kept"] == 2


def test_a_failed_prompt_still_stops_sampling(monkeypatch):
    step = Step(monkeypatch)
    step.dispatch("a", MIXED, 1)
    step.buffer.add("train", {"b": {"global_steps": 7, "status": "failure"}})
    with pytest.raises(RuntimeError, match=r"1 prompts failed in the agent loop at global_steps=7 \(first: b\)"):
        step.sample(num_groups=2)


def _trainer(filter_groups, adv_estimator="grpo", loss_mode="vanilla", data=None):
    trainer = object.__new__(main_ppo_sync.PPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"adv_estimator": adv_estimator, "filter_groups": filter_groups},
            "actor_rollout_ref": {"actor": {"policy_loss": {"loss_mode": loss_mode}}},
            "data": data or {"train_batch_size": 64},
        }
    )
    return trainer


def test_a_step_that_sends_more_prompts_than_it_keeps_needs_no_refill(monkeypatch):
    step = Step(monkeypatch, refills=[MIXED])
    for uid, scores in zip("abcdef", (FAIL, MIXED, PASS, MIXED, MIXED, MIXED), strict=True):
        step.dispatch(uid, scores, 1)
    kept, dropped, metrics = step.sample(num_groups=3, max_prompts=3)
    assert step.calls == []
    assert _uids(kept) == ["b", "d", "e"] and _uids(dropped) == ["a", "c", "f"]
    assert metrics["training/refill/surplus_varied_groups"] == 1


def test_refill_settings_follow_algorithm_filter_groups():
    assert _trainer(None)._refill_settings() is None
    assert _trainer({"enable": False, "metric": "reward"})._refill_settings() is None
    on = {"enable": True, "metric": "reward", "max_num_gen_batches": 3}
    assert _trainer(on)._refill_settings() == {"max_num_gen_batches": 3}
    with pytest.raises(ValueError, match="needs algorithm.adv_estimator=grpo"):
        _trainer(on, adv_estimator="gae")._refill_settings()
    with pytest.raises(ValueError, match="does not apply to the sdpo loss"):
        _trainer(on, loss_mode="sdpo")._refill_settings()
    with pytest.raises(ValueError, match="metric must be 'reward'"):
        _trainer({"enable": True, "metric": "acc"})._refill_settings()
    assert _trainer(on, data={"train_batch_size": 64, "gen_batch_size": 96})._refill_settings()
    with pytest.raises(ValueError, match="data.gen_batch_size=32 sends fewer prompts than the 64 groups"):
        _trainer(on, data={"train_batch_size": 64, "gen_batch_size": 32})._refill_settings()


def test_refill_prompts_come_from_the_same_pass_as_the_step_batches(monkeypatch):
    monkeypatch.setattr(
        main_ppo_sync,
        "tu",
        SimpleNamespace(get_tensordict=dict, assign_non_tensor_data=lambda b, k, v: b.update({k: v})),
    )
    trainer = object.__new__(main_ppo_sync.PPOTrainer)
    trainer._refill, trainer._step_prompts, trainer.global_steps = {"max_num_gen_batches": 0}, 2, 4
    trainer.train_dataloader = [[{"raw_prompt": f"p{i}", "input_ids": torch.tensor([i])}] for i in range(5)]
    sent = []
    trainer.async_rollout_manager = SimpleNamespace(generate_sequences=sent.append)

    batches = trainer._train_batches()
    first = next(batches)
    assert list(first["raw_prompt"]) == ["p0", "p1"] and first["input_ids"].tolist() == [[0], [1]]
    assert trainer._refill_prompts(1) == 1
    assert list(sent[0]["raw_prompt"]) == ["p2"] and len(sent[0]["uid"]) == 1 and sent[0]["global_steps"] == 4
    assert sent[0]["untraced"] is True
    assert list(next(batches)["raw_prompt"]) == ["p3", "p4"]
    assert trainer._refill_prompts(1) == 0
    assert next(batches, None) is None


def test_small_dispatches_rotate_over_the_workers_and_every_owner_is_kept(monkeypatch):
    monkeypatch.setattr(main_ppo_sync.ray, "get", lambda refs: refs)
    got = defaultdict(list)
    workers = [
        SimpleNamespace(generate_sequences=SimpleNamespace(remote=lambda chunk, w=w: got[w].extend(chunk["uid"])))
        for w in range(3)
    ]
    step = Step(monkeypatch)
    manager = object.__new__(main_ppo_sync.AgentLoopManagerTQ)
    manager.agent_loop_workers, manager.replay_buffer, manager._next_worker = workers, step.buffer, 0
    for uids in (["a", "b", "c"], ["d"], ["e"], ["f"]):
        batch = main_ppo_sync.tu.get_tensordict({"uid": np.array(uids, dtype=object)})
        main_ppo_sync.tu.assign_non_tensor_data(batch, "global_steps", 7)
        manager.generate_sequences(batch)
    assert dict(got) == {0: ["a", "d"], 1: ["b", "e"], 2: ["c", "f"]}
    assert step.buffer.owners["train"] == {
        "a": workers[0],
        "b": workers[1],
        "c": workers[2],
        "d": workers[0],
        "e": workers[1],
        "f": workers[2],
    }
