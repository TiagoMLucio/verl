# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Synchronous PPO trainer with colocated actor and rollout.

Differs from original PPO trainer in main_ppo.py:
1. Use TransferQueue to zero-padding and zero-copy data transfer.
2. Use ReplayBuffer to sample data from TransferQueue.
3. Support different `n` sampling for each prompt.
4. Support multiple outputs for each agent loop.
"""

import asyncio
import json
import logging
import math
import os
import threading
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pprint import pprint
from typing import Any

import hydra
import numpy as np
import ray
import torch

try:
    import transfer_queue as tq
    from transfer_queue import KVBatchMeta
except ImportError:
    print("Please install TQ by calling `pip install TransferQueue==0.1.6` and try again.")
    from verl.utils.transferqueue_utils import KVBatchMeta, tq

from omegaconf import DictConfig, OmegaConf, open_dict
from tensordict import NonTensorData, NonTensorStack, TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    get_trajectory_info,
)
from verl.experimental.reward_loop import RewardLoopManager
from verl.experimental.teacher_loop import MultiTeacherModelManager
from verl.protocol import DataProto, DataProtoFuture
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayWorkerGroup,
    ResourcePoolManager,
    create_colocated_worker_cls,
)
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler, run_ppo
from verl.trainer.ppo import core_algos, sdpo_teacher
from verl.trainer.ppo.core_algos import agg_loss, finalize_ratio_metrics
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.padding_utils import upsample_batch_to_divisible_size
from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_advantage, compute_spec_decode_metrics
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_teacher_policy
from verl.utils import hf_processor, hf_tokenizer
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.debug import marked_timer
from verl.utils.debug.metrics import calculate_debug_metrics
from verl.utils.device import auto_set_device
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.ray_utils import auto_await
from verl.utils.rollout_trace import RolloutTraceConfig
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.tensordict_utils import list_of_dict_to_tensordict
from verl.utils.tracking import Tracking, ValidationGenerationsLogger
from verl.workers.config import CriticConfig, DistillationConfig
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker, TrainingWorkerConfig
from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.utils.losses import value_loss
from verl.workers.utils.padding import response_from_nested, response_to_nested


def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
    params["top_p"] = 1.0
    params["top_k"] = -1
    params["temperature"] = 0


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


# ======================================= USER SECTION BEGIN =======================================


def compute_advantage_for_multi_trajectories(
    data: DataProto,
    batch_keys: list[str],
    adv_estimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Any = None,
) -> DataProto:
    """Compute GRPO advantages from each session's final output. For non-GRPO
    estimators, such as GAE, are delegated to the original compute_advantage() unchanged.

    For GRPO, only the final output in each ``{uid}_{session_id}`` group participates
    in advantage computation, and the result is broadcast to the other outputs in
    the same session. Sessions whose AgentLoop returns ``None`` simply do not appear
    in ``batch_keys``. Non-GRPO estimators, such as GAE, are delegated to the
    original ``compute_advantage()`` unchanged.
    """
    if adv_estimator != core_algos.AdvantageEstimator.GRPO:
        return compute_advantage(
            data,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

    # final session of each agent loop: {uid}_{session_id} => (index, row_index)
    final_sessions: dict[str, tuple[int, int]] = {}
    row_session_keys = []
    for i, key in enumerate(batch_keys):
        fields = key.rsplit("_", 2)
        assert len(fields) == 3, f"Unexpected key format: {key}"
        uid, session_id, index = fields[0], fields[1], int(fields[2])
        session_key = f"{uid}_{session_id}"
        row_session_keys.append(session_key)
        if session_key not in final_sessions or final_sessions[session_key][0] < index:
            final_sessions[session_key] = (index, i)

    # final session indices in batch data
    final_indices = []
    session_key_to_local_index = {}
    for session_key, (_, row_index) in final_sessions.items():
        final_indices.append(row_index)
        session_key_to_local_index[session_key] = len(final_indices) - 1
    row_to_local_index = [session_key_to_local_index[session_key] for session_key in row_session_keys]

    # select final sessions from batch data for group relative advantage computation
    final_data = compute_advantage(
        data.select_idxs(final_indices),
        adv_estimator=adv_estimator,
        gamma=gamma,
        lam=lam,
        num_repeat=num_repeat,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )
    first_nnz_indices = final_data.batch["response_mask"].argmax(dim=1)
    final_scores = final_data.batch["advantages"][torch.arange(len(final_data)), first_nnz_indices]

    # scatter final scores to all rows in batch data
    scores = final_scores[row_to_local_index]
    scores = scores.unsqueeze(-1) * data.batch["response_mask"]

    data.batch["advantages"] = scores
    data.batch["returns"] = scores
    return data


def _final_segment_local_indices(keys: list[str]) -> list[int]:
    """Row index of each session's final segment, from {uid}_{session_id}_{index} keys."""
    best: dict[str, tuple[int, int]] = {}
    for row, key in enumerate(keys):
        uid, session_id, index = key.rsplit("_", 2)
        session_key = f"{uid}_{session_id}"
        if session_key not in best or best[session_key][0] < int(index):
            best[session_key] = (int(index), row)
    return [row for _, row in best.values()]


class ReplayBuffer:
    """Replay buffer periodically polls metadata from transfer queue.

    Args:
        poll_interval (float, optional): Poll interval in seconds. Defaults to 1.0.
    """

    def __init__(self, poll_interval: float = 1.0):
        # partition_id => {key: tags}
        self.partitions: dict[str, dict[str, dict]] = defaultdict(dict)

        self.poll_interval = poll_interval
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self.poll_thread = threading.Thread(target=self._poll_from_transfer_queue, daemon=True)
        self.poll_thread.start()

    def _poll_from_transfer_queue(self):
        """Periodically poll metadata from transfer queue."""
        try:
            while not self._stop_event.is_set():
                data = tq.kv_list()
                if data is not None:
                    for partition_id, items in data.items():
                        self.add(partition_id, items)
                self._stop_event.wait(self.poll_interval)
        except Exception as e:
            if not self._stop_event.is_set():
                logger.error(f"Error in _poll_from_transfer_queue: {e}")
                os._exit(1)

    def close(self):
        """Stop the background polling thread."""
        if not self.poll_thread.is_alive():
            return
        self._stop_event.set()
        self.poll_thread.join(timeout=self.poll_interval + 1.0)
        if self.poll_thread.is_alive():
            logger.warning("ReplayBuffer poll thread did not stop within timeout")

    def add(self, partition_id: str, items: dict[str, dict]):
        """Add items to the replay buffer.

        Args:
            partition_id (str): Partition of transfer queue, e.g. "train" or "val".
            items (dict[str, dict]): Items to add, e.g. {"key": {"tag": "value"}}.
        """
        with self.lock:
            partition = self.partitions[partition_id]
            for key, tags in items.items():
                if key not in partition:
                    partition[key] = {}
                partition[key].update(tags)

    def remove(self, partition_id: str, keys: list[str]):
        """Remove items from the replay buffer.

        Args:
            partition_id (str): Partition of transfer queue, e.g. "train" or "val".
            keys (list[str]): Keys to remove.
        """
        with self.lock:
            partition = self.partitions[partition_id]
            for key in keys:
                if key in partition:
                    del partition[key]

    def sample(self, partition_id: str, global_steps: int = None, batch_size: int = None) -> KVBatchMeta:
        """Sample a batch of data from the replay buffer.

        Args:
            partition_id (str): Partition of transfer queue, e.g. "train" or "val".
            global_steps (int, optional): Global training steps. If not None, wait until all prompts of
                this global steps have finished.
            batch_size (int, optional): Batch size. Defaults to None.

        Returns:
            KVBatchMeta: A batch of data.
        """
        assert (global_steps is not None or batch_size) and (not (global_steps is not None and batch_size)), (
            "Either global_steps or batch_size must be specified, but not both."
        )

        while True:
            time.sleep(self.poll_interval)
            with self.lock:
                keys, tags = [], []
                should_wait = False
                partition = self.partitions[partition_id]
                for key, tag in partition.items():
                    if tag["global_steps"] == global_steps:
                        if tag["status"] == "running":
                            should_wait = True
                            break
                        elif tag["status"] == "success":
                            keys.append(key)
                            tags.append(tag)
                        else:
                            logger.debug(f"Unknown status {tag['status']} for key {key}")
                if not should_wait:
                    return KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags)


@ray.remote
class AgentLoopWorkerTQ(AgentLoopWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tq.init()
        self.background_tasks = set()

    async def generate_sequences(self, batch: TensorDict) -> None:
        """Spawn agent loop for each sample in the batch without waiting for the results."""
        from verl.utils.debug_breakpoints import should_break
        if should_break("agent_loop"): breakpoint()

        validate = batch["validate"] if "validate" in batch else False
        batch.pop("validate", None)
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch:
            default_agent_loop = config.agent.default_agent_loop
            batch["agent_name"] = NonTensorData(default_agent_loop)

        trajectory_info = await get_trajectory_info(batch["global_steps"], batch["index"], validate)

        # Select which samples to trace this step (mirrors AgentLoopWorker.generate_sequences):
        # trace up to max_samples_per_step_per_worker unique samples per worker (None traces all).
        raw_index = batch["index"]
        index = list(raw_index.tolist()) if hasattr(raw_index, "tolist") else list(raw_index)
        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker
        unique_indices = list(set(index))
        if max_samples_per_worker is not None and max_samples_per_worker < len(unique_indices):
            selected_samples = set(np.random.choice(unique_indices, max_samples_per_worker, replace=False).tolist())
            traced_indices = {i for i in range(len(batch)) if index[i] in selected_samples}
        else:
            traced_indices = set(range(len(batch)))

        # create background tasks for each sample in the batch
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            prompt = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    prompt[k] = v[i]
                elif isinstance(v, NonTensorStack):
                    prompt[k] = v[i].data
                elif isinstance(v, NonTensorData):
                    prompt[k] = v.data
                else:
                    logger.exception(f"Unsupported type {type(v)} for key {k}")

            # “fire-and-forget” background tasks
            task = asyncio.create_task(
                self._run_prompt(prompt, sampling_params, trajectory=trajectory_info[i], trace=trace_this_sample)
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    async def _run_prompt(self, prompt: dict, sampling_params: dict, trajectory: dict, trace: bool = False) -> None:
        """Spawn multiple agent loops in parallel according to rollout.n or rollout.val_kwargs.n."""
        uid, partition_id = prompt["uid"], "train" if not trajectory["validate"] else "val"
        try:
            # NOTE: user can dynamically adjust n for each sample here, e.g according to task difficulty.
            config = self.config.actor_rollout_ref.rollout
            n = prompt.pop("__rollout_n__", config.n if not trajectory["validate"] else config.val_kwargs.n)
            do_sample = prompt.pop("__do_sample__", True)

            run_sampling_params = dict(sampling_params)
            if not trajectory["validate"] and not do_sample:
                apply_greedy_sampling_params(run_sampling_params)

            tasks = []
            for i in range(n):
                task = asyncio.create_task(
                    self._run_agent_loop(
                        run_sampling_params,
                        trajectory=trajectory,
                        trace=trace,
                        session_id=i,
                        validate=trajectory["validate"],
                        **prompt,
                    )
                )
                tasks.append(task)
            await asyncio.gather(*tasks)
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "finished"})
        except Exception as e:
            logger.exception(f"Error in _run_prompt: {e}")
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})

    async def _agent_loop_postprocess(
        self, output: AgentLoopOutput | list[AgentLoopOutput], validate, **kwargs
    ) -> None:
        """Put agent loop outputs into TransferQueue."""
        from verl.utils.debug_breakpoints import should_break
        if should_break("postprocess"): breakpoint()
        uid, session_id = kwargs["uid"], kwargs["session_id"]
        outputs = output if isinstance(output, list) else [output]
        if not outputs:
            logger.warning(f"Empty output for prompt {uid}_{session_id}")
            return

        await self._compute_score(outputs, kwargs=kwargs)

        final_output = outputs[-1]
        # TODO: Support output:list[AgentLoopOutput]
        await self._compute_teacher_logprobs(
            final_output,
            prompt_ids=final_output.prompt_ids,
            response_ids=final_output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )

        if final_output.reward_score is not None:
            final_reward_extra_info = final_output.extra_fields.get("reward_extra_info")
            for output in outputs[:-1]:
                output.reward_score = final_output.reward_score
                if final_reward_extra_info is not None:
                    output.extra_fields["reward_extra_info"] = final_reward_extra_info

        # NOTE: agent loop may has multiple outputs, put each output into TransferQueue.
        # key format: {uid}_{session_id}_{index}
        # - uid: raw prompt uid from dataset
        # - session_id: session id for rollout.n sampling
        # - index: index of agent loop output
        keys, fields, tags = [], [], []
        for i, output in enumerate(outputs):
            prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
            responses = torch.tensor(output.response_ids, dtype=torch.int64)
            input_ids = torch.cat([prompts, responses], dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
            ).squeeze(0)

            keys.append(f"{uid}_{session_id}_{i}")
            field = output.as_dict()
            field.update(kwargs)
            # do not store raw image/video
            field.pop("multi_modal_data", None)
            # TODO: uniform response_mask and loss_mask
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["position_ids"] = position_ids
            field["multi_modal_inputs"] = multi_modal_inputs
            fields.append(field)
            prompt_len, response_len = field["prompts"].size(0), field["responses"].size(0)
            tags.append(
                {
                    "global_steps": kwargs["global_steps"],
                    "status": "success",
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "seq_len": prompt_len + response_len,
                }
            )

        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id="train" if not validate else "val",
        )


class AgentLoopManagerTQ(AgentLoopManager):
    def __init__(self, *args, replay_buffer: ReplayBuffer, **kwargs):
        self.agent_loop_workers_class = AgentLoopWorkerTQ
        super().__init__(*args, **kwargs)
        self.replay_buffer = replay_buffer

    @classmethod
    @auto_await
    async def create(
        cls,
        *args,
        replay_buffer: ReplayBuffer = None,
        **kwargs,
    ):
        """Create agent loop manager."""
        instance = cls(
            *args,
            **kwargs,
            replay_buffer=replay_buffer,
        )
        await instance._init_agent_loop_workers()
        return instance

    def generate_sequences(self, prompts: TensorDict) -> None:
        """
        Dispatch input batch to agent loop workers without blocking. Workers should put agent loop outputs
        into TransferQueue once an agent loop finished.

        Args:
            prompts (TensorDict): Input batch from train or validation dataset.
        """
        # mark prompts as pending in replay buffer
        global_steps = prompts["global_steps"]
        partition_id = "train" if "validate" not in prompts else "val"
        items = {uid: {"global_steps": global_steps, "status": "running"} for uid in prompts["uid"]}
        self.replay_buffer.add(partition_id, items)

        chunkes = prompts.chunk(len(self.agent_loop_workers))
        ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=False)
            ]
        )


# ======================================= USER SECTION END =======================================


class PPOTrainer:
    """PPO Trainer with TransferQueue and ReplayBuffer.

    Args:
        config: DictConfig from yaml config file.
        role_worker_mapping: dict[Role, WorkerType]
        resource_pool_manager: ResourcePoolManager
    """

    def __init__(
        self,
        config: DictConfig,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
    ):
        self.config = config
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_critic = need_critic(self.config)
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_teacher_policy = need_teacher_policy(self.config)
        self.replay_buffer = ReplayBuffer()
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._init_tokenizer()
        self._init_dataloader()
        self._init_dump_executor()

    def _init_tokenizer(self):
        """Initialize tokenizer."""
        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            self.config.actor_rollout_ref.model.path, use_shm=self.config.actor_rollout_ref.model.get("use_shm", False)
        )
        trust_remote_code = self.config.data.get("trust_remote_code", False)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if loss_mode == "sdpo":
            self.tokenizer.padding_side = "left"
            reprompt_truncation = self.config.actor_rollout_ref.actor.get("self_distillation", {}).get(
                "reprompt_truncation"
            )
            if reprompt_truncation in {"left", "right"}:
                self.tokenizer.truncation_side = reprompt_truncation
        # Used for multimodal LLM, could be None
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    def _init_dataloader(self):
        """Initialize train and validate dataloader."""
        self.train_dataset = create_rl_dataset(
            self.config.data.train_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            is_train=True,
            max_samples=self.config.data.get("train_max_samples", -1),
        )
        self.val_dataset = create_rl_dataset(
            self.config.data.val_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            is_train=False,
            max_samples=self.config.data.get("val_max_samples", -1),
        )

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data["dataloader_num_workers"],
            drop_last=True,
            collate_fn=collate_fn,
            sampler=create_rl_sampler(self.config.data, self.train_dataset),
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.val_batch_size or len(self.val_dataset),
            num_workers=self.config.data["dataloader_num_workers"],
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )
        logger.info(
            f"train and validate dataloader initialized, train dataset size: "
            f"{len(self.train_dataset)}, val dataset size: {len(self.val_dataset)}"
        )

        # adjust total_training_steps
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        logger.info(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            logger.warning(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        from verl.utils.debug_breakpoints import should_break
        if should_break("init_workers"): breakpoint()

        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # 1. define actor and rollout class
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[actor_role],
            config=self.config.actor_rollout_ref,
            distillation_config=self.config.get("distillation"),
            role=str(actor_role),
        )
        self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls

        # 2. define critic class
        if self.use_critic:
            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)
            critic_cfg.engine.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            critic_cfg.engine.max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            worker_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=critic_cfg.model_config,
                engine_config=critic_cfg.engine,
                optimizer_config=critic_cfg.optim,
                checkpoint_config=critic_cfg.checkpoint,
            )
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=worker_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # 3. create worker group for actor rollout and critic
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.config.trainer.device
        logger.info(f"worker group kwargs: {wg_kwargs}")

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = RayWorkerGroup(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            logger.info(f"create worker group {spawn_wg.keys()}")

        # 5. initialize critic model engine
        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.reset()
            value_loss_ = partial(value_loss, config=critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)
            logger.info("critic model engine initialized")

        # 6. initialize actor and ref model engine
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()
        logger.info("actor and ref model engine initialized")

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = self.config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = self.config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or self.config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        if self.use_reference_policy:
            self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # 7. initialize reward loop manager
        resource_pool = (
            self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            if self.config.reward.reward_model.enable
            else None
        )
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )
        logger.info("reward loop manager initialized")

        # 8. initialize teacher loop manager
        if self.use_teacher_policy:
            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
            self.teacher_model_manager = MultiTeacherModelManager(
                config=self.config,
                resource_pool=teacher_resource_pool,
            )
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.teacher_model_manager = None
            self.distillation_config = None

        # 9. initialize agent loop manager
        self.llm_server_manager = LLMServerManager.create(
            config=self.config, worker_group=self.actor_rollout_wg, rollout_resource_pool=actor_rollout_resource_pool
        )

        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            agent_loop_manager_cls = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            agent_loop_manager_cls = AgentLoopManagerTQ
        self.async_rollout_manager = agent_loop_manager_cls.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(),
            teacher_client=self.teacher_model_manager.get_client() if self.use_teacher_policy else None,
            reward_loop_worker_handles=self.reward_loop_manager.reward_loop_workers,
            replay_buffer=self.replay_buffer,
        )
        logger.info("agent loop manager initialized")

        # 10. initialize checkpoint engine manager
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )
        logger.info("checkpoint engine manager initialized")

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

        logger.info("all initialize finished, ready to fit")

    def _load_checkpoint(self):
        self.global_steps = 0

        # 1. find latest checkpoint folder
        if self.config.trainer.resume_mode == "disable":
            return
        elif self.config.trainer.resume_mode == "auto":
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest
            if global_step_folder is None:
                logger.info("Training from scratch")
                return
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
            assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)
        else:
            logger.exception(f"Unknown resume mode {self.config.trainer.resume_mode}")

        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        logger.info(f"Resuming from {global_step_folder}, setting global step to {self.global_steps}")

        # 2. load actor checkpoint
        self.actor_rollout_wg.load_checkpoint(
            local_path=os.path.join(global_step_folder, "actor"),
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )

        # 3. load critic checkpoint
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                local_path=os.path.join(global_step_folder, str(Role.Critic)),
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )

        # 4. load dataloader checkpoint
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            logger.warning(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _save_checkpoint(self):
        """Save actor, critic, and dataloader checkpoints to local (and optionally remote) storage."""
        from verl.utils.fs import local_mkdir_safe

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        logger.info(f"Saving checkpoint to {local_global_step_folder}")

        # resolve max checkpoints to keep
        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            logger.warning(
                "remove_previous_ckpt_in_save is deprecated, "
                "set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        # save actor
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        # save critic
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader state
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        torch.save(self.train_dataloader.state_dict(), dataloader_local_path)

        # write latest checkpointed iteration tracker for atomic resume
        actor_ckpt_cfg = self.config.actor_rollout_ref.actor.get("checkpoint", {})
        if actor_ckpt_cfg.get("async_save", False):
            logger.info("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _validate(self) -> dict[str, float]:
        # Lists to collect samples for the table
        sample_uids = []
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_exit_reasons = []
        data_sources = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        dump_all_inputs: list[str] = []
        dump_all_outputs: list[str] = []
        dump_all_keys: list[str] = []
        dump_all_indices: list = []
        session_to_sample_idx: dict[str, int] = {}

        for batch_dict in self.val_dataloader:
            # 1. put batch to agent loop manager
            n_prompts = len(batch_dict["raw_prompt"])
            val_n = self.config.actor_rollout_ref.rollout.val_kwargs.n
            logger.info(f"validation dispatch: {n_prompts} prompts x n={val_n} = {n_prompts * val_n} agent loops")
            batch_dict["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object
            )
            batch = tu.get_tensordict(batch_dict)
            tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
            tu.assign_non_tensor_data(batch, "validate", True)
            self.async_rollout_manager.generate_sequences(batch)

            # 2. sample batch from replay buffer
            batch = self.replay_buffer.sample(partition_id="val", global_steps=self.global_steps)

            # 3. [OPTIONAL] compute reward score with colocated reward model
            if self.reward_loop_manager.reward_loop_worker_handles is None:
                self.checkpoint_manager.sleep_replicas()
                batch = self._compute_reward_colocate(batch)
                self.checkpoint_manager.update_weights()

            # 4. collect necessary data for logging
            # For multi-output agent loops, only use the final output per session for metrics.
            # Keys have format {uid}_{session_id}_{index}; keep only the highest index per session.
            session_max: dict[str, tuple[int, int]] = {}  # session_key -> (max_index, position)
            for pos, key in enumerate(batch.keys):
                parts = key.rsplit("_", 2)
                if len(parts) == 3:
                    session_key = f"{parts[0]}_{parts[1]}"
                    index = int(parts[2])
                    if session_key not in session_max or index > session_max[session_key][0]:
                        session_max[session_key] = (index, pos)
                else:
                    session_max[key] = (0, pos)
            sorted_sessions = sorted(session_max.items(), key=lambda x: x[1][1])
            final_indices = [pos for _, (_, pos) in sorted_sessions]
            final_keys = [batch.keys[i] for i in final_indices]
            base_offset = len(sample_scores)
            session_to_sample_idx.update(
                {session_key: base_offset + j for j, (session_key, _) in enumerate(sorted_sessions)}
            )

            text_data = tq.kv_batch_get(
                keys=batch.keys, partition_id=batch.partition_id, select_fields=["prompts", "responses"]
            )
            text_data["prompts"] = text_data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            text_data["responses"] = text_data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            all_inputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in text_data["prompts"]]
            all_outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in text_data["responses"]]

            fields = ["uid", "rm_scores", "num_turns", "reward_model", "data_source", "extra_fields"]
            data = tq.kv_batch_get(keys=final_keys, partition_id=batch.partition_id, select_fields=fields)

            sample_uids.extend(data.pop("uid").tolist())
            sample_outputs.extend(all_outputs[i] for i in final_indices)
            sample_inputs.extend(all_inputs[i] for i in final_indices)
            scores = data["rm_scores"].sum(dim=1).tolist()
            sample_scores.extend(scores)
            sample_turns.extend(data.pop("num_turns").tolist())
            reward_extra_infos_dict["reward"].extend(scores)

            extra_fields_list = data.pop("extra_fields", None)
            if extra_fields_list is None:
                sample_exit_reasons.extend([None] * len(scores))
            if extra_fields_list is not None:
                n_prior = len(reward_extra_infos_dict["reward"]) - len(extra_fields_list.tolist())
                for extra_field in extra_fields_list.tolist():
                    # how a val rollout ended lives at the top level of extra_fields, next to
                    # reward_extra_info rather than inside it; without it a dumped val
                    # trajectory cannot say whether it submitted, ran out of turns or got
                    # stuck, which is the whole comparison between arms. Dump-only: the
                    # value is a string and process_validation_metrics aggregates numerics.
                    sample_exit_reasons.append(
                        extra_field.get("traj_exit_reason") if isinstance(extra_field, dict) else None
                    )
                    reward_extra_info = (
                        extra_field.get("reward_extra_info", {}) if isinstance(extra_field, dict) else {}
                    )
                    for key in reward_extra_infos_dict:
                        if key != "reward" and key not in reward_extra_info:
                            reward_extra_infos_dict[key].append(None)
                    for key, value in reward_extra_info.items():
                        if key not in reward_extra_infos_dict:
                            reward_extra_infos_dict[key] = [None] * n_prior
                        reward_extra_infos_dict[key].append(value)
                    n_prior += 1

            reward_model = data.pop("reward_model", None)
            if reward_model is not None:
                sample_gts.extend([item.get("ground_truth", None) for item in reward_model.tolist()])
            else:
                sample_gts.extend([None] * len(final_indices))

            data_source = data.pop("data_source", None)
            if data_source is not None:
                data_sources.extend(data_source.tolist())
            else:
                data_sources.extend(["unknown"] * len(final_indices))

            dump_all_inputs.extend(all_inputs)
            dump_all_outputs.extend(all_outputs)
            dump_all_keys.extend(batch.keys)
            # read before the queue is cleared below
            batch_indices = self._fetch_sample_indices(batch)
            if batch_indices is None or len(batch_indices) != len(batch.keys):
                batch_indices = [None] * len(batch.keys)
            dump_all_indices.extend(batch_indices)

            # 5. cleanup transfer queue and replay buffer
            tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)
            self.replay_buffer.remove(batch.partition_id, batch.keys)

        # logger to wandb
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump to local dir
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            # Sort according to uid (so that generations in the same rollout are together)
            sort_keys = []
            for key in dump_all_keys:
                parts = key.rsplit("_", 2)
                sort_keys.append((parts[0], int(parts[1]), int(parts[2])) if len(parts) == 3 else (key, 0, 0))
            sorted_indices = sorted(range(len(dump_all_keys)), key=lambda i: sort_keys[i])
            dump_all_inputs = [dump_all_inputs[i] for i in sorted_indices]
            dump_all_outputs = [dump_all_outputs[i] for i in sorted_indices]
            dump_all_keys = [dump_all_keys[i] for i in sorted_indices]
            dump_all_indices = [dump_all_indices[i] for i in sorted_indices]

            # For ground truths, scores and reward extra infos, find the values in the
            # lists for the final samples of each session
            dump_all_sessions = [
                f"{parts[0]}_{parts[1]}" if len(parts) == 3 else key
                for key in dump_all_keys
                for parts in [key.rsplit("_", 2)]
            ]
            session_final_indices = [session_to_sample_idx[session] for session in dump_all_sessions]
            self._dump_generations(
                inputs=dump_all_inputs,
                outputs=dump_all_outputs,
                gts=[sample_gts[i] for i in session_final_indices],
                scores=[sample_scores[i] for i in session_final_indices],
                reward_extra_infos_dict={
                    k: [v[i] for i in session_final_indices] for k, v in reward_extra_infos_dict.items()
                }
                | {"uid": dump_all_keys}
                | ({"sample_index": dump_all_indices} if any(i is not None for i in dump_all_indices) else {})
                | (
                    {"traj_exit_reason": [sample_exit_reasons[i] for i in session_final_indices]}
                    if any(r is not None for r in sample_exit_reasons)
                    else {}
                ),
                dump_path=val_data_dir,
            )

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""
        generations_to_log = self.config.trainer.log_val_generations
        if generations_to_log == 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    @staticmethod
    def _write_generations(inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, global_steps):
        """Write generation samples as JSONL (runs in background thread)."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        def json_encode_default(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif hasattr(obj, "tolist"):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False, default=json_encode_default) + "\n")

        print(f"Dumped generations to {filename}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL asynchronously."""
        global_steps = self.global_steps
        future = self._dump_executor.submit(
            self._write_generations,
            inputs,
            outputs,
            gts,
            scores,
            reward_extra_infos_dict,
            dump_path,
            global_steps,
        )
        self._dump_futures.append(future)
        # Clean up completed futures and surface any exceptions early
        still_pending = []
        for f in self._dump_futures:
            if f.done():
                f.result()  # re-raises if the write failed
            else:
                still_pending.append(f)
        self._dump_futures = still_pending

    def _init_dump_executor(self):
        """Create or recreate the dump executor and futures list."""
        self._dump_executor = ThreadPoolExecutor(max_workers=1)
        self._dump_futures = []

    def _shutdown_dump_executor(self):
        """Drain pending dump futures and shut down the executor."""
        for f in self._dump_futures:
            f.result()
        self._dump_futures.clear()
        self._dump_executor.shutdown(wait=True)

    @staticmethod
    def _fetch_sample_indices(batch: KVBatchMeta):
        """Dataset row ids (`extra_info.index`) for a batch, or None if unavailable.

        Kept as its own read so a missing column degrades the dump instead of taking
        the training step down with it.
        """
        try:
            got = tq.kv_batch_get(
                keys=batch.keys, partition_id=batch.partition_id, select_fields=["index"]
            )["index"]
            return got.tolist() if hasattr(got, "tolist") else list(got)
        except Exception as e:  # noqa: BLE001 - dumping is best-effort
            logger.warning(f"sample_index unavailable for this dump: {e}")
            return None

    def _log_rollout_data(self, batch: KVBatchMeta, timing_raw: dict, rollout_data_dir: str):
        """Fetch rollout data from TransferQueue and dump sorted by uid."""
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            fields = ["uid", "prompts", "responses", "rm_scores", "reward_model", "extra_fields"]
            data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
            data["prompts"] = data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            data["responses"] = data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)

            uids = data.pop("uid").tolist()
            inputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in data["prompts"]]
            outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in data["responses"]]
            scores = data["rm_scores"].sum(dim=1).tolist()

            reward_model = data.pop("reward_model", None)
            if reward_model is not None:
                gts = [item.get("ground_truth", None) for item in reward_model.tolist()]
            else:
                gts = [None] * len(uids)

            # Sort by uid key ({sample}_{rollout}_{output})
            sort_keys = []
            for key in batch.keys:
                parts = key.rsplit("_", 2)
                if len(parts) == 3:
                    sort_keys.append((parts[0], int(parts[1]), int(parts[2])))
                else:
                    sort_keys.append((key, 0, 0))
            sorted_indices = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])

            inputs = [inputs[i] for i in sorted_indices]
            outputs = [outputs[i] for i in sorted_indices]
            gts = [gts[i] for i in sorted_indices]
            scores = [scores[i] for i in sorted_indices]

            reward_extra_infos_dict = {"uid": [batch.keys[i] for i in sorted_indices]}
            # rollout traces tag every span with sample_index; carrying it here is what
            # lets a dumped trajectory be tied back to its own timings
            sample_indices = self._fetch_sample_indices(batch)
            if sample_indices is not None and len(sample_indices) == len(batch.keys):
                reward_extra_infos_dict["sample_index"] = [sample_indices[i] for i in sorted_indices]
            # downstream hint analysis reads turn_feedback/turn_spans from the dump, not trace exports
            extra_fields = data.pop("extra_fields", None)
            if extra_fields is not None:
                ef = extra_fields.tolist()
                for key in ("turn_feedback", "turn_spans"):
                    reward_extra_infos_dict[key] = [
                        json.dumps((ef[i] or {}).get(key)) for i in sorted_indices
                    ]
                # how a trajectory ended is otherwise only an aggregate fraction in the
                # metrics: without it a dumped rollout cannot say whether it submitted,
                # ran out of turns or got stuck
                reward_extra_infos_dict["traj_exit_reason"] = [
                    (ef[i] or {}).get("traj_exit_reason") for i in sorted_indices
                ]

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=rollout_data_dir,
            )

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns) -> dict[str, float]:
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        # with multiple sources (e.g. difficulty bands) also log the combined total under "all"
        if len(set(data_sources)) > 1:
            merged = process_validation_metrics(["all"] * len(data_sources), sample_uids, reward_extra_infos_dict)
            data_src2var2metric2val.update(merged)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.array(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _start_profiling(self) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        do_profile = (
            not self.prev_step_profile and self.curr_step_profile
            if self.config.global_profiler.profile_continuous_steps
            else self.curr_step_profile
        )

        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        self.next_step_profile = (
            self.global_steps + 1 in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        do_profile = (
            self.curr_step_profile and not self.next_step_profile
            if self.config.global_profiler.profile_continuous_steps
            else self.curr_step_profile
        )
        self.prev_step_profile = self.curr_step_profile
        self.curr_step_profile = self.next_step_profile

        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _compute_reward_colocate(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the reward with colocate reward model."""
        # TODO: add reward model
        raise NotImplementedError

    def _add_remax_reward_baselines(self, batch: KVBatchMeta) -> KVBatchMeta:
        """Attach one greedy baseline reward to every sampled ReMax trajectory."""
        baseline_prefix = "remax_baseline_"
        sampled_keys, sampled_tags = [], []
        all_baseline_keys = []
        final_baseline_key_by_uid: dict[str, tuple[int, str]] = {}
        for key, tag in zip(batch.keys, batch.tags, strict=True):
            uid, _, index = key.rsplit("_", 2)
            if uid.startswith(baseline_prefix):
                all_baseline_keys.append(key)
                output_index = int(index)
                if uid not in final_baseline_key_by_uid or final_baseline_key_by_uid[uid][0] < output_index:
                    final_baseline_key_by_uid[uid] = (output_index, key)
            else:
                sampled_keys.append(key)
                sampled_tags.append(tag)

        assert final_baseline_key_by_uid, "ReMax requires greedy baseline rollout outputs, but none were found."
        baseline_keys = [key for _, key in final_baseline_key_by_uid.values()]
        baseline_data = tq.kv_batch_get(
            keys=baseline_keys, partition_id=batch.partition_id, select_fields=["uid", "rm_scores"]
        )
        baseline_scores = baseline_data["rm_scores"].sum(dim=-1)
        baseline_by_uid = {
            uid.removeprefix(baseline_prefix): score
            for uid, score in zip(list(baseline_data["uid"]), baseline_scores, strict=True)
        }

        sampled_data = tq.kv_batch_get(keys=sampled_keys, partition_id=batch.partition_id, select_fields=["uid"])
        reward_baselines = torch.stack([baseline_by_uid[uid] for uid in list(sampled_data["uid"])])
        tq.kv_batch_put(
            keys=sampled_keys,
            partition_id=batch.partition_id,
            fields=TensorDict({"reward_baselines": reward_baselines}, batch_size=len(sampled_keys)),
        )
        tq.kv_clear(keys=all_baseline_keys, partition_id=batch.partition_id)
        self.replay_buffer.remove(batch.partition_id, all_baseline_keys)
        return KVBatchMeta(
            keys=sampled_keys,
            tags=sampled_tags,
            partition_id=batch.partition_id,
            fields=batch.fields,
            extra_info=batch.extra_info,
        )

    def _maybe_build_self_distillation_batch(self, batch: KVBatchMeta, metrics: dict) -> None:
        """Build SDPO teacher inputs and distillation masks when loss_mode is set to ``sdpo``.

        Mirrors ``RayPPOTrainer._maybe_build_self_distillation_batch``, adapted to TransferQueue:
        the rollout outputs are read from TQ instead of a ``DataProto`` (``reward_tensor`` and
        ``reward_extra_infos_dict``, which the legacy trainer receives as arguments, are
        reconstructed here), and the per-sample teacher sequence (``teacher_input_ids`` = teacher
        prompt + response) and ``self_distillation_mask`` are written back to TQ instead of
        returned. The teacher attention mask / position ids are derived and recomputed inside the
        actor worker (see ``reconstruct_padded_teacher_from_nested``).

        With ``use_turn_feedback``, supervision is hints-only: samples carrying reflection hints
        ship one spliced teacher sequence (hints inserted before their turns, with
        ``teacher_seq_meta`` mapping hinted spans back to the response grid) and a per-token
        distillation mask over those spans; un-hinted samples are not trained at all (degenerate
        1-token teacher row, zero mask — the teacher still scores them so dp collectives stay
        in lockstep).
        """
        self_distillation_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if self_distillation_cfg is None or loss_mode != "sdpo":
            return
        from verl.utils.debug_breakpoints import should_break
        if should_break("teacher_build"): breakpoint()

        turn_mode = bool(self_distillation_cfg.get("use_turn_feedback", False))
        select_fields = ["responses", "rm_scores", "raw_prompt", "uid", "extra_fields", "response_mask"]
        if turn_mode:
            select_fields.append("prompts")
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=select_fields,
        )

        if "raw_prompt" not in data:
            raise ValueError("SDPO requires `raw_prompt` in TransferQueue to build teacher prompts.")
        if "uid" not in data:
            raise ValueError("SDPO requires `uid` in TransferQueue.")

        responses = data["responses"]
        batch_size = len(batch.keys)
        uids = list(data["uid"])
        reward_tensor = data["rm_scores"].to_padded_tensor(padding=0.0)
        reward_extra_infos_dict = {
            "feedback": [
                extra_field.get("reward_extra_info", {}).get("feedback") if isinstance(extra_field, dict) else None
                for extra_field in data["extra_fields"]
            ]
        }

        response_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in responses.unbind()]
        raw_prompts = list(data["raw_prompt"])

        prompt_texts = [sdpo_teacher.extract_prompt_text(raw_prompt) for raw_prompt in raw_prompts]

        feedback_list = sdpo_teacher.collect_feedback(
            include_environment_feedback=self_distillation_cfg.include_environment_feedback,
            reward_extra_infos_dict=reward_extra_infos_dict,
            batch_size=batch_size,
        )

        success_by_uid = sdpo_teacher.collect_solutions_by_uid(
            uids,
            reward_tensor,
            success_reward_threshold=self_distillation_cfg.success_reward_threshold,
        )
        extra_fields_list = list(data["extra_fields"])
        contexts = [
            sdpo_teacher.TeacherSampleContext(
                raw_prompt=raw_prompts[i],
                prompt_text=prompt_texts[i],
                solution=sdpo_teacher.select_solution(
                    i,
                    success_by_uid,
                    uids,
                    response_texts,
                    self_distillation_cfg.dont_reprompt_on_self_success,
                    self_distillation_cfg.get("remove_thinking_from_demonstration", False),
                ),
                feedback=feedback_list[i],
                extra_fields=extra_fields_list[i] if isinstance(extra_fields_list[i], dict) else {},
            )
            for i in range(batch_size)
        ]

        # Turn mode is hints-only: hinted samples get a spliced per-sample teacher sequence, the
        # rest a degenerate 1-token row (untrained). Non-turn mode keeps the legacy reprompt context.
        response_list = list(responses.unbind())
        response_mask_list = list(data["response_mask"].unbind())
        hinted_per_row = [
            sdpo_teacher.select_hinted_turns(
                ctx.extra_fields, response_list[i].shape[0], self_distillation_cfg.get("max_hinted_turns")
            )
            if turn_mode
            else []
            for i, ctx in enumerate(contexts)
        ]
        # Hints-only in turn mode: un-hinted rows are not trained, so no reprompt context is built.
        reprompt_rows = [] if turn_mode else list(range(batch_size))

        messages = [sdpo_teacher.build_teacher_messages(contexts[i], self_distillation_cfg) for i in reprompt_rows]
        stripped_prompts: dict[int, torch.Tensor] = {}
        if messages:
            apply_kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}))
            chat_template_kwargs = dict(
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
                max_length=self_distillation_cfg.max_reprompt_len,
                padding=True,
                truncation=True,
                **apply_kwargs,
            )
            try:
                teacher_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    continue_final_message=False,
                    **chat_template_kwargs,
                )
            except TypeError:
                teacher_prompt = self.tokenizer.apply_chat_template(messages, **chat_template_kwargs)

            if isinstance(teacher_prompt, torch.Tensor):
                teacher_prompt_input_ids = teacher_prompt
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                teacher_prompt_attention_mask = (teacher_prompt_input_ids != pad_token_id).to(dtype=torch.long)
            else:
                teacher_prompt_input_ids = teacher_prompt["input_ids"]
                teacher_prompt_attention_mask = teacher_prompt.get("attention_mask")
                if teacher_prompt_attention_mask is None:
                    teacher_prompt_attention_mask = torch.ones_like(teacher_prompt_input_ids, dtype=torch.long)
            stripped_prompts = {
                row: teacher_prompt_input_ids[j][teacher_prompt_attention_mask[j].bool()]
                for j, row in enumerate(reprompt_rows)
            }

        feedback_used = [sdpo_teacher.feedback_used(ctx, self_distillation_cfg) for ctx in contexts]
        legacy_mask = [ctx.solution is not None or used for ctx, used in zip(contexts, feedback_used, strict=True)]

        # The legacy trainer builds teacher_input_ids = cat(teacher_prompt, responses) together with
        # the teacher attention mask and position ids. Here we store only the no-padding teacher
        # sequence per sample (the teacher prompt, stripped of its left padding, followed by the real
        # response tokens); the actor worker re-pads it and recomputes the mask / position ids.
        if turn_mode:
            # meta [n_sub, (total_len, body_len, body_start, start, end) per sub-row] maps spliced positions back to the response grid
            prompt_list = data["prompts"].unbind()
            hint_template = self_distillation_cfg.turn_feedback_template
            call_template = self_distillation_cfg.call_feedback_template
            template_kwargs = dict(getattr(self_distillation_cfg, "chat_template_kwargs", None) or {})
            header_ids = torch.tensor(
                sdpo_teacher.assistant_header_ids(self.tokenizer, template_kwargs=template_kwargs), dtype=torch.int64
            )
            # mid-turn (at == "call") splices close the assistant turn and reopen it after the
            # hint; the call span starts at the template's tool-call opening token
            close_ids = torch.tensor(
                self.tokenizer.encode(self.tokenizer.eos_token + "\n", add_special_tokens=False), dtype=torch.int64
            )
            call_open_ids = torch.tensor(
                self.tokenizer.encode("<tool_call>", add_special_tokens=False), dtype=torch.int64
            )
            encode = lambda t: self.tokenizer.encode(t, add_special_tokens=False)

            # call_target=forced: splice the corrected call into the student's own row, so the
            # update recomputes its log-probs under the corrected prefix and distills the
            # teacher there. Runs before old_log_prob (step 5), so every downstream consumer
            # sees the modified rows consistently. Sibling per-position rows are spliced in
            # lockstep; the modified fields ship back to TQ with the teacher fields below.
            forced_rows: dict[int, dict] = {}
            rollout_lp_list = None
            if self_distillation_cfg.call_target == "forced" and any(hinted_per_row):
                call_close_ids = torch.tensor(
                    self.tokenizer.encode("</tool_call>", add_special_tokens=False), dtype=torch.int64
                )
                extra_sel = ["position_ids"]
                if self.config.actor_rollout_ref.rollout.calculate_log_probs:
                    extra_sel.append("rollout_log_probs")
                row_data = tq.kv_batch_get(
                    keys=batch.keys, partition_id=batch.partition_id, select_fields=extra_sel
                )
                if "rollout_log_probs" in row_data:
                    rollout_lp_list = list(row_data["rollout_log_probs"].unbind())
                # rebuilt position ids are a plain arange: only valid for 1-D (text) rope
                text_rope = row_data["position_ids"].unbind()[0].dim() == 1
                for i in range(batch_size):
                    if not (hinted_per_row[i] and text_rope):
                        continue
                    swap = sdpo_teacher.forced_call_swap(
                        response_list[i], hinted_per_row[i], encode, call_open_ids,
                        call_close_ids, self_distillation_cfg.call_mask)
                    # the swapped region must be model-generated: never put supervised loss
                    # where the response mask protects observation tokens
                    if swap is None or int(
                        response_mask_list[i][swap["at"]: swap["at"] + swap["removed"]].min()
                    ) != 1:
                        continue
                    response_list[i] = swap["response_ids"]
                    hinted_per_row[i] = swap["hinted_turns"]
                    response_mask_list[i] = sdpo_teacher.splice_row(
                        response_mask_list[i], swap["at"], swap["removed"],
                        torch.ones(swap["inserted"]))
                    if rollout_lp_list is not None:
                        # marked, not zeroed: a zero reads as log p = 0, and the rollout-correction
                        # weight exp(old_lp - rollout_lp) would then mute the forced tokens
                        rollout_lp_list[i] = sdpo_teacher.splice_row(
                            rollout_lp_list[i], swap["at"], swap["removed"],
                            torch.full((swap["inserted"],), sdpo_teacher.FORCED_LP_MARKER))
                    forced_rows[i] = swap

            teacher_seqs, seq_meta, mask_rows = [], [], []
            target_rows = [] if self_distillation_cfg.call_target == "onehot" else None
            hint_fallbacks = 0
            from verl.utils.debug_breakpoints import should_break
            for i in range(batch_size):
                response_ids = response_list[i]
                if hinted_per_row[i]:
                    if should_break("teacher_build"): breakpoint()
                    hint_ids = [
                        torch.tensor(
                            sdpo_teacher.hint_user_turn_ids(
                                self.tokenizer,
                                (call_template if at == "call" else hint_template).format(diagnosis=text),
                                template_kwargs=template_kwargs,
                            ),
                            dtype=response_ids.dtype,
                        )
                        for *_, text, at, _target in hinted_per_row[i]
                    ]
                    seq, meta, fallbacks, spans, call_placed = sdpo_teacher.build_spliced_teacher_row(
                        prompt_list[i],
                        response_ids,
                        hinted_per_row[i],
                        hint_ids,
                        # the spliced prefix is the student's real prompt (segment rows reach ~24k):
                        # cap with the student's own prompt budget, not the legacy reprompt one
                        self.config.data.max_prompt_length,
                        header_ids,
                        close_ids=close_ids,
                        call_open_ids=call_open_ids,
                    )
                    hint_fallbacks += fallbacks
                    # narrow target-bearing at-call spans to the tokens the corrected call
                    # changes; the loss denominator follows the mask, so this is where the
                    # copy-token dilution is actually removed
                    if target_rows is not None:
                        target_rows.append(sdpo_teacher.call_target_rows(
                            spans, call_placed, hinted_per_row[i], response_ids, encode,
                            response_ids.shape[0]))
                    if i in forced_rows:
                        # the swapped call already IS the target: re-narrowing would diff
                        # identical sequences and keep the full span. Its supervised spans
                        # (diffed against the original call, on the corrected grid)
                        # substitute; the row's other call hints are narrowed as usual.
                        fr = forced_rows[i]
                        keep = [j for j in range(len(spans)) if j != fr["hint_idx"]]
                        spans = sdpo_teacher.narrowed_call_spans(
                            [spans[j] for j in keep], [call_placed[j] for j in keep],
                            [hinted_per_row[i][j] for j in keep], response_ids, encode,
                            self_distillation_cfg.call_mask,
                            decode_fn=self.tokenizer.decode)
                        spans += fr["mask_spans"]
                    else:
                        spans = sdpo_teacher.narrowed_call_spans(
                            spans, call_placed, hinted_per_row[i], response_ids, encode,
                            self_distillation_cfg.call_mask,
                            decode_fn=self.tokenizer.decode)
                    mask_row = sdpo_teacher.turn_token_mask(response_ids.shape[0], spans)
                else:
                    # hints-only: degenerate 1-token teacher row (padding-template pattern), zero mask.
                    # The teacher must still score every row so dp-group collectives stay in lockstep.
                    seq = torch.cat([prompt_list[i][-1:], response_ids[:1]])
                    meta = [1, 2, 1, 0, 0, 1]  # one sub-row over the 2-token stub
                    mask_row = torch.zeros(response_ids.shape[0], dtype=torch.float32)
                    if target_rows is not None:
                        target_rows.append(torch.full((response_ids.shape[0],), -1, dtype=torch.int64))
                teacher_seqs.append(seq)
                seq_meta.append(torch.tensor(meta, dtype=torch.int64))
                mask_rows.append(mask_row)
            teacher_fields = {
                "teacher_input_ids": torch.nested.nested_tensor(teacher_seqs, layout=torch.jagged),
                "teacher_seq_meta": torch.nested.nested_tensor(seq_meta, layout=torch.jagged),
                "self_distillation_mask": torch.nested.nested_tensor(mask_rows, layout=torch.jagged),
            }
            if target_rows is not None:
                teacher_fields["call_target_ids"] = torch.nested.nested_tensor(target_rows, layout=torch.jagged)
        else:
            teacher_input_ids = torch.nested.nested_tensor(
                [torch.cat([stripped_prompts[i], response_list[i]]) for i in range(batch_size)],
                layout=torch.jagged,
            )
            teacher_fields = {
                "teacher_input_ids": teacher_input_ids,
                "self_distillation_mask": torch.tensor(legacy_mask, dtype=torch.float32),
            }

        # loss_mask = supervised mask: batch_num_tokens all-reduces to the global supervised-token count
        if turn_mode:
            loss_mask_rows = [
                response_mask_list[i] * mask_rows[i].to(response_mask_list[i].dtype) for i in range(batch_size)
            ]
        else:
            loss_mask_rows = [response_mask_list[i] * int(legacy_mask[i]) for i in range(batch_size)]
        teacher_fields["loss_mask"] = torch.nested.nested_tensor(loss_mask_rows, layout=torch.jagged)

        # Forced swaps change row contents and lengths: ship the modified rows and every
        # per-position sibling back to TQ so old_log_prob, advantages, and the update all
        # see one consistent grid. Unchanged rows are rewritten identically.
        if turn_mode and forced_rows:
            seqs = [torch.cat([prompt_list[i], response_list[i]]) for i in range(batch_size)]
            teacher_fields["responses"] = torch.nested.nested_tensor(response_list, layout=torch.jagged)
            teacher_fields["response_mask"] = torch.nested.nested_tensor(
                response_mask_list, layout=torch.jagged)
            teacher_fields["input_ids"] = torch.nested.nested_tensor(seqs, layout=torch.jagged)
            teacher_fields["attention_mask"] = torch.nested.nested_tensor(
                [torch.ones_like(s) for s in seqs], layout=torch.jagged)
            teacher_fields["position_ids"] = torch.nested.nested_tensor(
                [torch.arange(s.shape[0], dtype=torch.int64) for s in seqs], layout=torch.jagged)
            if rollout_lp_list is not None:
                teacher_fields["rollout_log_probs"] = torch.nested.nested_tensor(
                    rollout_lp_list, layout=torch.jagged)
            metrics.update(
                {
                    "self_distillation/forced_swapped_rows": float(len(forced_rows)),
                    "self_distillation/forced_skipped_rows": float(
                        sum(1 for h in hinted_per_row if h) - len(forced_rows)
                    ),
                    "self_distillation/forced_delta_tokens": float(
                        sum(fr["inserted"] - fr["removed"] for fr in forced_rows.values())
                    ),
                }
            )

        # A condensed trajectory ships one row per segment; segment_index==0 marks it once so
        # fractions count trajectories rather than segments (long, failing traces split most).
        num_segments_per_row = [
            max(1, int((ef or {}).get("num_segments", 1) or 1)) if isinstance(ef, dict) else 1
            for ef in extra_fields_list
        ]
        first_seg = [
            i
            for i, ef in enumerate(extra_fields_list)
            if not isinstance(ef, dict) or int(ef.get("segment_index", 0) or 0) == 0
        ]
        # A trajectory counts once in total, and its segments split that weight by how much
        # supervision each carries: the loss is then a token-mean within a trajectory and a plain
        # mean across trajectories. Splitting evenly instead would over-weight a segment that only
        # holds one short hinted turn. Keys are '{sample}_{session}_{segment}'.
        supervised_per_row = [float(mask.sum()) for mask in loss_mask_rows]
        traj_of_row = [tuple(key.rsplit("_", 2)[:2]) for key in batch.keys]
        n_traces = len(set(traj_of_row))
        # call-hinted rows carry ~10x the per-token divergence of turn-hinted ones, so they
        # dominate the update at equal row weight; call_loss_weight rebalances the two channels
        call_row = [any(hint[4] == "call" for hint in hinted_per_row[i]) for i in range(batch_size)]
        # Renormalized to the supervised-row count: raw shares sum to the number of supervised
        # trajectories, which would otherwise shrink the update by the average segments-per-
        # trajectory (~0.6x at our condensation rate) and confound comparisons across runs.
        weights = sdpo_teacher.trace_weights(
            supervised_per_row, traj_of_row, call_row,
            self_distillation_cfg.call_loss_weight if turn_mode else 1.0,
        )
        teacher_fields["trace_weight"] = torch.tensor(weights, dtype=torch.float32).unsqueeze(-1)
        # Row -> trajectory, as a plain int the update path can carry: mini-batches are cut
        # from shuffled rows, so a condensed trajectory's supervised segments land in
        # different optimizer steps even though their weights are a single trajectory's share.
        traj_ids = {traj: i for i, traj in enumerate(dict.fromkeys(traj_of_row))}
        teacher_fields["traj_id"] = torch.tensor(
            [traj_ids[traj] for traj in traj_of_row], dtype=torch.int64
        ).unsqueeze(-1)

        segs_per_traj = Counter(traj_of_row)
        sup_segs_per_traj = Counter(traj for traj, n in zip(traj_of_row, supervised_per_row) if n > 0)
        unsup_rows = [i for i, n in enumerate(supervised_per_row) if n == 0]
        metrics.update(
            {
                "self_distillation/rows_per_step": float(batch_size),
                "self_distillation/traces_per_step": float(n_traces),
                "self_distillation/segments_per_trace_max": float(max(segs_per_traj.values(), default=0)),
                "self_distillation/supervised_segments_per_trace_max": float(
                    max(sup_segs_per_traj.values(), default=0)
                ),
                # rows that run a full student forward+backward for zero gradient
                "self_distillation/unsupervised_row_fraction": len(unsup_rows) / max(batch_size, 1),
                "self_distillation/unsupervised_row_tokens": float(
                    sum(int(response_mask_list[i].sum()) for i in unsup_rows)
                ),
                "self_distillation/supervised_row_tokens": float(
                    sum(int(response_mask_list[i].sum()) for i in range(batch_size) if supervised_per_row[i] > 0)
                ),
            }
        )

        unique_uids = set(uids)
        num_with_feedback_available = sum(1 for i in first_seg if feedback_list[i] is not None)
        num_with_feedback_used = sum(1 for i in first_seg if feedback_used[i])
        num_with_solution = sum(1 for i in first_seg if contexts[i].solution is not None)
        num_supervised = sum(
            1 for i in range(batch_size) if hinted_per_row[i] or (not turn_mode and legacy_mask[i])
        )
        metrics.update(
            {
                "self_distillation/success_group_fraction": len(
                    [uid for uid in unique_uids if len(success_by_uid[uid]) > 0]
                )
                / len(unique_uids),
                "self_distillation/success_sample_fraction": num_with_solution / n_traces,
                "self_distillation/feedback_available_fraction": num_with_feedback_available / n_traces,
                "self_distillation/feedback_used_fraction": num_with_feedback_used / n_traces,
                "self_distillation/reprompt_sample_fraction": num_supervised / batch_size,
            }
        )
        metrics.update(
            self._condensation_metrics(
                extra_fields_list, num_segments_per_row, reward_tensor, traj_of_row,
                self_distillation_cfg.success_reward_threshold,
            )
        )
        metrics.update(self._trajectory_timing_metrics(extra_fields_list))
        # decode throughput needs the generated count: response_length also holds the observations
        # fed back to the agent, which are ~3x the tokens the model actually produced
        generated = sum(
            sum(int(span[2]) - int(span[1]) for span in (ef if isinstance(ef, dict) else {}).get("turn_spans") or [])
            for ef in extra_fields_list
        )
        metrics["rollout/generated_tokens"] = float(generated)
        metrics["rollout/generated_tokens_per_trace"] = generated / n_traces if n_traces else 0.0
        if turn_mode:
            num_hinted = sum(1 for hinted in hinted_per_row if hinted)
            hinted_traces = {traj_of_row[i] for i, hinted in enumerate(hinted_per_row) if hinted}
            metrics.update(
                {
                    "self_distillation/hinted_sample_fraction": num_hinted / batch_size,
                    "self_distillation/hinted_trace_fraction": len(hinted_traces) / n_traces,
                    "self_distillation/hinted_turns_per_sample": (
                        sum(len(hinted) for hinted in hinted_per_row) / num_hinted if num_hinted else 0.0
                    ),
                    "self_distillation/hinted_turns_per_trace": (
                        sum(len(hinted) for hinted in hinted_per_row) / len(hinted_traces) if hinted_traces else 0.0
                    ),
                    "self_distillation/hint_injection_fallbacks": hint_fallbacks,
                    # the two supervision channels, as the loss actually weighs them
                    "self_distillation/call_row_fraction": (
                        sum(1 for i, c in enumerate(call_row) if c and supervised_per_row[i] > 0)
                        / max(sum(1 for n in supervised_per_row if n > 0), 1)
                    ),
                    "self_distillation/call_row_weight_share": (
                        sum(w for w, c in zip(weights, call_row, strict=True) if c)
                        / max(sum(weights), 1e-8)
                    ),
                    "self_distillation/call_loss_weight": float(
                        self_distillation_cfg.call_loss_weight
                    ),
                }
            )
            metrics.update(self._hint_position_metrics(hinted_per_row, extra_fields_list, traj_of_row))

        tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=TensorDict(teacher_fields, batch_size=batch_size),
        )

    @staticmethod
    def _condensation_metrics(
        extra_fields_list, num_segments_per_row, reward_tensor, traj_of_row, success_threshold
    ) -> dict:
        """Condensation reach and whether it predicts the outcome.

        Rows are segments; a trajectory is its segment_index==0 row, and all its segments carry
        the same reward, so per-trace stats read off the first-segment rows.
        """
        seq_scores = reward_tensor.sum(dim=-1).detach().cpu().tolist()
        traces, turns_by_seg = [], defaultdict(list)
        for i, ef in enumerate(extra_fields_list):
            ef = ef if isinstance(ef, dict) else {}
            seg_idx = int(ef.get("segment_index", 0) or 0)
            spans = ef.get("turn_spans") or []
            if spans:  # failed rollouts ship empty rows; counting them as 0 turns skews the mean
                turns_by_seg[min(seg_idx, 3)].append(len(spans))
            if seg_idx == 0:
                traces.append((num_segments_per_row[i], seq_scores[i], ef.get("traj_exit_reason")))
        if not traces:
            return {}
        n = len(traces)
        solved = lambda score: score >= success_threshold  # noqa: E731
        out = {
            "rollout/condensed_trace_fraction": sum(1 for s, _, _ in traces if s > 1) / n,
            "rollout/segments_per_trace": sum(s for s, _, _ in traces) / n,
        }
        for bucket in (1, 2, 3):
            sel = [sc for s, sc, _ in traces if (s == bucket if bucket < 3 else s >= 3)]
            name = f"{bucket}seg" if bucket < 3 else "3plusseg"
            if sel:
                out[f"rollout/solve_rate_{name}"] = sum(1 for sc in sel if solved(sc)) / len(sel)
                out[f"rollout/trace_fraction_{name}"] = len(sel) / n
        reasons = [r for _, _, r in traces if r]
        for reason in set(reasons):
            sub = [sc for _, sc, r in traces if r == reason]
            out[f"rollout/exit_{reason}_fraction"] = len(sub) / n
            out[f"rollout/solve_rate_exit_{reason}"] = sum(1 for sc in sub if solved(sc)) / len(sub)
        for seg_idx, counts in sorted(turns_by_seg.items()):
            name = str(seg_idx) if seg_idx < 3 else "3plus"
            out[f"rollout/turns_in_segment_{name}"] = sum(counts) / len(counts)
        return out

    @staticmethod
    def _trajectory_timing_metrics(extra_fields_list) -> dict:
        """Per-trajectory time split. The residual (loop_wall minus the parts) is the in-loop
        overhead we have not attributed yet; step wall clock is set by the slowest trajectory,
        so the max matters more than the mean."""
        rows = [
            (ef or {}).get("timings") or {}
            for ef in extra_fields_list
            if isinstance(ef, dict) and int((ef or {}).get("segment_index", 0) or 0) == 0
        ]
        rows = [t for t in rows if t.get("loop_wall")]
        if not rows:
            return {}
        parts = ("generate_sequences", "tool_calls", "condense", "parse_action", "tokenize_observations")
        out = {}
        for key in parts + ("loop_wall", "env_setup", "reward_eval", "reflect"):
            vals = [float(t.get(key, 0.0)) for t in rows]
            out[f"traj_time/{key}_mean"] = sum(vals) / len(vals)
        residual = [
            max(0.0, float(t.get("loop_wall", 0.0)) - sum(float(t.get(k, 0.0)) for k in parts)) for t in rows
        ]
        totals = [
            float(t.get("loop_wall", 0.0))
            + float(t.get("env_setup", 0.0))
            + float(t.get("reward_eval", 0.0))
            + float(t.get("reflect", 0.0))
            for t in rows
        ]
        # vLLM preemption count per trajectory: >0 means the KV cache could not hold the
        # working set, so sequences were evicted and their prefill recomputed (wasted GPU
        # work that shows up as high utilization with low goodput). -1 = engine did not report.
        preempted = [float(t.get("num_preempted", -1)) for t in rows]
        reported = [p for p in preempted if p >= 0]
        out["rollout/preempted_reported_fraction"] = len(reported) / len(preempted)
        if reported:
            out["rollout/preempted_mean"] = sum(reported) / len(reported)
            out["rollout/preempted_max"] = max(reported)
            out["rollout/preempted_trace_fraction"] = sum(1 for p in reported if p > 0) / len(reported)
        out["traj_time/unattributed_mean"] = sum(residual) / len(residual)
        out["traj_time/total_mean"] = sum(totals) / len(totals)
        out["traj_time/total_max"] = max(totals)
        # the spread between these is the idle tail: the phase lasts as long as the max
        ordered = sorted(totals)
        out["traj_time/total_p50"] = ordered[len(ordered) // 2]
        out["traj_time/total_p90"] = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
        # the slowest trajectory sets the step's wall clock, so its own split is what matters
        slowest = rows[max(range(len(rows)), key=lambda i: totals[i])]
        for key in parts + ("loop_wall", "env_setup", "reward_eval", "reflect"):
            out[f"traj_time/slowest_{key}"] = float(slowest.get(key, 0.0))
        out["traj_time/unattributed_share"] = sum(residual) / max(sum(totals), 1e-6)
        for key in ("eval_completed", "patch_apply_failed", "empty_patch", "reflect_failed", "reflect_empty"):
            vals = [float(t[key]) for t in rows if key in t]  # absent means never measured, not OK
            if vals:
                out[f"reward_health/{key}_fraction"] = sum(vals) / len(vals)
        capped = [float(t.get("capped_turns", 0.0)) for t in rows]
        out["reward_health/capped_turns_mean"] = sum(capped) / len(capped)
        out["reward_health/capped_rollouts_fraction"] = sum(1.0 for c in capped if c > 0) / len(capped)
        return out

    @staticmethod
    def _hint_position_metrics(hinted_per_row, extra_fields_list, traj_of_row) -> dict:
        """Where in a trajectory the hints land: late hints supervise turns nothing can still fix.

        Step ranges are pooled per trajectory, since a condensed trace splits its turns across
        rows and a per-row range would call every segment-final hint a last-turn hint.
        """
        traj_steps = defaultdict(list)
        for traj, ef in zip(traj_of_row, extra_fields_list, strict=True):
            spans = (ef if isinstance(ef, dict) else {}).get("turn_spans") or []
            traj_steps[traj].extend(int(span[0]) for span in spans)
        rel, gaps, last_two = [], [], 0
        for hinted, traj in zip(hinted_per_row, traj_of_row, strict=True):
            steps = sorted(traj_steps.get(traj) or [0])
            lo, hi = steps[0], steps[-1]
            span_len = max(hi - lo, 1)
            hinted_steps = sorted(int(step) for step, *_ in hinted)
            rel.extend((step - lo) / span_len for step in hinted_steps)
            gaps.extend(b - a for a, b in zip(hinted_steps, hinted_steps[1:]))
            last_two += sum(1 for step in hinted_steps if step >= hi - 1)
        if not rel:
            return {}
        srt = sorted(rel)
        return {
            "self_distillation/hint_position_mean": sum(rel) / len(rel),
            "self_distillation/hint_position_median": srt[len(srt) // 2],
            "self_distillation/hint_position_first_half": sum(1 for r in rel if r <= 0.5) / len(rel),
            "self_distillation/hint_in_last_two_turns": last_two / len(rel),
            "self_distillation/hint_gap_mean": (sum(gaps) / len(gaps)) if gaps else 0.0,
            "self_distillation/hint_adjacent_fraction": (
                (sum(1 for g in gaps if g <= 2) / len(gaps)) if gaps else 0.0
            ),
        }

    def _get_required_batch_multiple(self, dp_size: int) -> int:
        """Return the global batch multiple required by downstream train steps(e.g. critics, actors)."""
        required_multiple = dp_size

        # If enabled with critic training, the batch should align with critic PPO mini-batches.
        if self.use_critic:
            critic_global_mini_batch_size = self.config.critic.ppo_mini_batch_size
            critic_global_mini_batch_size *= self.config.actor_rollout_ref.rollout.n
            required_multiple = math.lcm(required_multiple, critic_global_mini_batch_size)

        # If there is an actor update, the batch should align with actor PPO mini-batches too.
        if self.config.trainer.critic_warmup <= self.global_steps:
            actor_global_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            actor_global_mini_batch_size *= self.config.actor_rollout_ref.rollout.n
            required_multiple = math.lcm(required_multiple, actor_global_mini_batch_size)

        # Notice lcm(a, b, c) == lcm(lcm(a, b), c), so it is optimal.
        return required_multiple

    def _balance_batch(self, batch: KVBatchMeta, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens."""
        # get actor dp size
        role, worker_group = "actor", self.actor_rollout_wg
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        dp_size = max(dp_rank_mapping) + 1

        # Upsampling the batch with padding sequences
        batch_multiple = self._get_required_batch_multiple(dp_size)
        batch = upsample_batch_to_divisible_size(batch, batch_multiple, self.tokenizer.eos_token_id)
        global_seqlen_lst = torch.tensor([tag["seq_len"] for tag in batch.tags], dtype=torch.int64)
        workload_lst = calculate_workload(global_seqlen_lst)

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        batch.reorder([j for partition in global_partition_lst for j in partition])
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)
        return batch

    def _compute_old_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the old log prob of the batch."""
        # Operating Mode Selection:
        # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
        # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
        #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
            data = tq.kv_batch_get(
                keys=batch.keys, partition_id=batch.partition_id, select_fields=["rollout_log_probs"]
            )
            data["old_log_probs"] = data.pop("rollout_log_probs")
            tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data)
            return

        # 1. compute log probs
        batch.extra_info.update(
            {
                "calculate_entropy": True,
                "compute_loss": False,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            }
        )
        output: KVBatchMeta = self.actor_rollout_wg.compute_log_prob(batch)
        assert len(output) == len(batch)

        fields = ["entropy", "log_probs", "response_mask"]
        if self.config.actor_rollout_ref.rollout.calculate_log_probs:
            fields.extend(["responses", "rollout_log_probs"])
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        # 2. write old_log_probs and entropy back to TransferQueue
        data["old_log_probs"] = response_from_nested(data.pop("log_probs"), data["response_mask"])
        data["entropy"] = response_from_nested(data.pop("entropy"), data["response_mask"])
        batch = tq.kv_batch_put(
            keys=batch.keys, partition_id=batch.partition_id, fields=data.select("old_log_probs", "entropy")
        )

        data = DataProto(batch=data.to_padded_tensor())
        if "rollout_log_probs" in data.batch:
            data.batch["rollout_log_probs"] = sdpo_teacher.restore_forced_rollout_lp(
                data.batch["rollout_log_probs"], data.batch["old_log_probs"]
            )

        # 3. calculate actor entroy metrics
        actor_config = self.config.actor_rollout_ref.actor
        entropy_agg = agg_loss(
            loss_mat=data.batch["entropy"],
            loss_mask=data.batch["response_mask"],
            loss_agg_mode=actor_config.loss_agg_mode,
            loss_scale_factor=actor_config.loss_scale_factor,
        )
        old_log_prob_metrics = {
            "actor/entropy": entropy_agg.detach().item(),
            # "perf/mfu/actor_infer": old_log_prob_mfu,
        }
        metrics.update(old_log_prob_metrics)

        # 4. calculate rollout vs actor logprobs diff
        if self.config.actor_rollout_ref.rollout.calculate_log_probs:
            metrics.update(calculate_debug_metrics(data))

        return batch

    def _compute_ref_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the reference log prob of the batch."""
        # 1. compute log probs
        metadata = {
            "calculate_entropy": False,
            "compute_loss": False,
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
        }
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        batch.extra_info.update(metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.compute_log_prob(batch)
        else:
            output = self.ref_policy_wg.compute_ref_log_prob(batch)
        assert len(output) == len(batch)

        # 2. write ref_log_prob and entropy back to TransferQueue
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["log_probs", "response_mask"]
        )
        data["ref_log_prob"] = response_from_nested(data.pop("log_probs"), data["response_mask"])
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data.select("ref_log_prob"))

        return batch

    def _compute_values(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the values of the batch."""
        # 1. compute value
        output = self.critic_wg.infer_batch(batch)
        # TODO: DataProtoFuture support KVBatchMeta
        ray.get(output.futures)

        # 2. write value back to TransferQueue
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["values", "response_mask"]
        )
        data["values"] = response_from_nested(data.pop("values"), data["response_mask"])
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data.select("values"))

        return batch

    def _compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the advantage of the batch."""
        fields = ["uid", "response_mask", "rm_scores", "rollout_log_probs", "old_log_probs", "ref_log_prob", "values"]
        if self.config.algorithm.adv_estimator == core_algos.AdvantageEstimator.REMAX:
            fields.append("reward_baselines")
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        response_mask = data["response_mask"]
        data = DataProto(batch=data.to_padded_tensor())
        data.batch["token_level_scores"] = data.batch["rm_scores"]
        data.non_tensor_batch["uid"] = np.array(data.batch.pop("uid").tolist(), dtype=object)
        if "rollout_log_probs" in data.batch:
            data.batch["rollout_log_probs"] = sdpo_teacher.restore_forced_rollout_lp(
                data.batch["rollout_log_probs"], data.batch["old_log_probs"]
            )

        # 1. apply kl penalty to rewards
        if self.config.algorithm.use_kl_in_reward:
            data, kl_metrics = apply_kl_penalty(
                data, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
            )
            metrics.update(kl_metrics)
        else:
            data.batch["token_level_rewards"] = data.batch["token_level_scores"]

        # 2. Compute rollout correction: IS weights, rejection sampling, and metrics
        # Only runs in decoupled mode (computes once per batch using stable π_old)
        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        rollout_correction = (
            rollout_corr_config is not None and "rollout_log_probs" in data.batch and not bypass_recomputing_logprobs
        )
        if rollout_correction:
            data, is_metrics = compute_rollout_correction_and_add_to_batch(data, rollout_corr_config)
            metrics.update(is_metrics)

        # 3. compute advantages
        data = compute_advantage_for_multi_trajectories(
            data,
            batch_keys=batch.keys,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=self.config.algorithm,
        )

        # 4. write nested advantages and returns back to TransferQueue
        fields = ["advantages", "returns"]
        if self.config.algorithm.use_kl_in_reward:
            fields.append("token_level_rewards")
        if rollout_correction:
            fields.append("response_mask")
            if "rollout_is_weights" in data.batch:
                fields.append("rollout_is_weights")

        output = {}
        for field in fields:
            output[field] = response_to_nested(data.batch[field], response_mask)
        output = TensorDict(output, batch_size=len(batch))

        batch = tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)

        return batch

    def _update_critic(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Update the critic network."""
        ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        extra_info = {
            "global_batch_size": ppo_mini_batch_size,
            "mini_batch_size": ppo_mini_batch_size,
            "epochs": self.config.critic.ppo_epochs,
            "seed": self.config.critic.data_loader_seed,
            "dataloader_kwargs": {"shuffle": self.config.critic.shuffle},
        }
        batch.extra_info.update(extra_info)

        output: DataProtoFuture = self.critic_wg.train_mini_batch(batch)
        output: TensorDict = output.get()
        output = rename_dict(output["metrics"], "critic/")
        output["perf/mfu/critic"] = output.pop("critic/mfu")
        critic_metrics = reduce_metrics(output)
        metrics.update(critic_metrics)

        return batch

    def _drop_unsupervised_rows(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Keep only rows the update can learn from, then re-pad for divisibility.

        A row whose trace_weight is zero contributes no gradient: seq-mean-token-mean drops
        fully masked sequences and weights the rest by that number. It still costs a full
        forward and backward, which is a third of the update at our hint rate.

        The per-trajectory weighting is untouched. A row with no supervision has share
        zero, so neither traj_supervised nor the renormalising scale moves when it goes.
        """
        weights = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["trace_weight"]
        )["trace_weight"]
        weights = (weights.to_padded_tensor(0.0) if weights.is_nested else weights).reshape(len(batch.keys), -1)
        supervised = [bool(w.abs().sum() > 0) for w in weights.unbind()]
        kept = sum(supervised)
        metrics["self_distillation/dropped_unsupervised_rows"] = len(supervised) - kept
        if kept == 0 or kept == len(supervised):
            return batch

        batch = KVBatchMeta(
            keys=[k for k, keep in zip(batch.keys, supervised, strict=True) if keep],
            tags=[t for t, keep in zip(batch.tags, supervised, strict=True) if keep],
            partition_id=batch.partition_id,
            fields=batch.fields,
            extra_info=batch.extra_info,
        )
        role, worker_group = "actor", self.actor_rollout_wg
        if role not in worker_group._dispatch_info:
            worker_group._dispatch_info[role] = worker_group._query_dispatch_info(role)
        dp_size = max(worker_group._dispatch_info[role]) + 1
        return upsample_batch_to_divisible_size(
            batch, self._get_required_batch_multiple(dp_size), self.tokenizer.eos_token_id
        )

    def _update_actor(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Update the actor network."""
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        self_distillation_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        sdpo_enabled = self_distillation_cfg is not None and loss_mode == "sdpo"
        sdpo_needs_logits_processor = sdpo_enabled and self_distillation_cfg.get("full_logit_distillation", False)
        distillation_use_topk = sdpo_needs_logits_processor or (
            self.distillation_config.distillation_loss.loss_settings.use_topk
            if is_distillation_enabled(self.config.get("distillation"))
            else False
        )
        extra_info = {
            "calculate_entropy": calculate_entropy,
            "distillation_use_topk": distillation_use_topk,
            "global_batch_size": ppo_mini_batch_size,
            "mini_batch_size": ppo_mini_batch_size,
            "epochs": self.config.actor_rollout_ref.actor.ppo_epochs,
            "seed": self.config.actor_rollout_ref.actor.data_loader_seed,
            "dataloader_kwargs": {"shuffle": self.config.actor_rollout_ref.actor.shuffle},
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
        }
        # a separate handle: this function's return feeds _compute_metrics, which must still
        # see every row or reward and length statistics would describe the supervised subset
        update_batch = batch
        if self.config.actor_rollout_ref.actor.get("drop_unsupervised_rows", False):
            update_batch = self._drop_unsupervised_rows(batch, metrics)
        update_batch.extra_info.update(extra_info)

        output: TensorDict = self.actor_rollout_wg.update_actor(update_batch)
        output = rename_dict(output["metrics"], "actor/")
        output["perf/mfu/actor"] = output.pop("actor/mfu")
        # after reduce_metrics: the summed pairs are plain floats here, Metric objects before it
        actor_metrics = finalize_ratio_metrics(reduce_metrics(output), prefix="actor/")
        metrics.update(actor_metrics)

        return batch

    def _compute_metrics(self, batch: KVBatchMeta, metrics, timing_raw, global_steps, epoch):
        # 1. collect necessary fields from TransferQueue for computing metrics
        non_padding_mask = np.array([not tag.get("is_padding", False) for tag in batch.tags], dtype=bool)
        # One row per trajectory = each session's final segment. Trajectory-level metrics
        # (reward, num_turns) must use these, else they are weighted by segment count.
        mb_keys = [k for k, keep in zip(batch.keys, non_padding_mask) if keep]
        final_seg_idx = _final_segment_local_indices(mb_keys)
        multi_segment = 0 < len(final_seg_idx) < len(mb_keys)
        fields = [
            "prompts",
            "responses",
            "response_mask",
            "values",
            "advantages",
            "returns",
            "rm_scores",
            "token_level_rewards",
            "num_turns",
        ]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        num_turns = np.array(data.pop("num_turns").tolist())
        prompt_length = data["prompts"].offsets().diff()
        response_length = data["responses"].offsets().diff()
        global_token_num = (prompt_length + response_length).tolist()

        # Only fetch speculative decoding stats when rollout writes them.
        spec_drafts = spec_accepts = spec_verifies = None
        mtp_config = getattr(self.config.actor_rollout_ref.model, "mtp", None)
        if mtp_config is not None and mtp_config.enable and mtp_config.enable_rollout:
            spec_data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["extra_fields"],
            )
            extra_fields = spec_data["extra_fields"].tolist()
            spec_drafts = [extra_field["spec_num_draft_tokens"] for extra_field in extra_fields]
            spec_accepts = [extra_field["spec_num_accepted_tokens"] for extra_field in extra_fields]
            spec_verifies = [extra_field["spec_num_verify_steps"] for extra_field in extra_fields]

        data = data.to_padded_tensor()
        data["token_level_scores"] = data["rm_scores"]
        if "token_level_rewards" not in data:
            data["token_level_rewards"] = data["rm_scores"]
        data["prompt_length"] = prompt_length.float()
        data["response_length"] = response_length.float()
        batch = DataProto(batch=data, meta_info={"global_token_num": global_token_num})
        metrics_batch = batch.select_idxs(non_padding_mask) if non_padding_mask.any() else batch

        # 2. compute metrics
        metrics.update({"training/global_step": global_steps, "training/epoch": epoch})
        metrics.update(compute_data_metrics(batch=metrics_batch, use_critic=self.use_critic))
        if multi_segment:
            # score/reward are per-trajectory; recompute their stats over final segments only.
            per_traj = compute_data_metrics(batch=metrics_batch.select_idxs(final_seg_idx), use_critic=self.use_critic)
            metrics.update(
                {k: v for k, v in per_traj.items() if k.startswith(("critic/score/", "critic/rewards/"))}
            )
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
        # decode throughput: only assistant spans are generated, so response_length (which counts
        # the observations the agent was fed) overstates it several-fold
        gen_tokens = metrics.get("rollout/generated_tokens")
        gen_s = timing_raw.get("gen")
        if gen_tokens and gen_s:
            metrics["perf/gen_tokens_per_s"] = gen_tokens / gen_s / max(n_gpus, 1)
        gradient_norm = metrics.get("actor/grad_norm", None)
        metrics.update(compute_variance_proxy_metrics(batch=metrics_batch, gradient_norm=gradient_norm))

        # 3. other auxiliary metrics
        if non_padding_mask.any():
            num_turns = num_turns[non_padding_mask]
        if multi_segment:
            num_turns = num_turns[final_seg_idx]  # per-trajectory (one per session)
        metrics.update(
            {
                "training/num_turns/mean": num_turns.mean(),
                "training/num_turns/max": num_turns.max(),
                "training/num_turns/min": num_turns.min(),
            }
        )

        # 4. per-request speculative-decoding aggregation (same metrics async PPO logs;
        # see compute_spec_decode_metrics in verl/trainer/ppo/ray_trainer.py).
        metrics.update(compute_spec_decode_metrics(spec_drafts, spec_accepts, spec_verifies, non_padding_mask))

    def fit(self):
        from verl.utils.debug_breakpoints import should_break
        if should_break("fit"): breakpoint()

        if self._dump_executor._shutdown:
            self._init_dump_executor()

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        # perform validation before training
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            self.logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._shutdown_dump_executor()
                return

        current_epoch = self.global_steps // len(self.train_dataloader)
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.prev_step_profile = False
        self.curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        self.next_step_profile = False

        last_val_metrics = None
        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                is_last_step = self.global_steps >= self.total_training_steps
                metrics, timing_raw = {}, {}

                # 1. perform rollout and actor/critic training
                self._start_profiling()
                with marked_timer("step", timing_raw):
                    batch = self.step(batch_dict, metrics, timing_raw)

                    # 2. save checkpoint
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    # 3. update weights from trainer to rollout
                    with marked_timer("update_weights", timing_raw, color="red"):
                        self.checkpoint_manager.update_weights()
                self._stop_profiling()

                # 4. validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # 5. record metrics
                self._compute_metrics(batch, metrics, timing_raw, global_steps=self.global_steps, epoch=epoch)

                # 6. dump rollout generations if enabled
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    self._log_rollout_data(batch, timing_raw, rollout_data_dir)

                # 7. cleanup transfer queue and replay buffer
                tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)
                self.replay_buffer.remove(batch.partition_id, batch.keys)

                self.logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    self._shutdown_dump_executor()
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    self.logger.finish()
                    return

        # Ensure dump executor is shut down when training loop ends without reaching is_last_step
        self._shutdown_dump_executor()
        self.logger.finish()

    def step(self, batch_dict: dict, metrics: dict, timing_raw: dict) -> KVBatchMeta:
        # 1. put batch to agent loop manager
        batch_dict["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object)
        if self.config.algorithm.adv_estimator == core_algos.AdvantageEstimator.REMAX:
            rollout_n = self.config.actor_rollout_ref.rollout.n
            sampled_batch_dict = batch_dict.copy()
            sampled_batch_dict["__do_sample__"] = np.ones(len(batch_dict["raw_prompt"]), dtype=bool)
            sampled_batch_dict["__rollout_n__"] = np.full(len(batch_dict["raw_prompt"]), rollout_n, dtype=np.int64)

            baseline_batch_dict = batch_dict.copy()
            baseline_batch_dict["uid"] = np.array([f"remax_baseline_{uid}" for uid in batch_dict["uid"]], dtype=object)
            baseline_batch_dict["__do_sample__"] = np.zeros(len(batch_dict["raw_prompt"]), dtype=bool)
            baseline_batch_dict["__rollout_n__"] = np.ones(len(batch_dict["raw_prompt"]), dtype=np.int64)

            batch = torch.cat([tu.get_tensordict(sampled_batch_dict), tu.get_tensordict(baseline_batch_dict)], dim=0)
        else:
            batch = tu.get_tensordict(batch_dict)
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        self.async_rollout_manager.generate_sequences(batch)

        # 2. sample batch from replay buffer
        with marked_timer("gen", timing_raw, color="red"):
            batch = self.replay_buffer.sample(partition_id="train", global_steps=self.global_steps)
        batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        self.checkpoint_manager.sleep_replicas()

        # 3. [OPTIONAL] compute reward score with colocated reward model
        if self.reward_loop_manager.reward_loop_worker_handles is None:
            with marked_timer("reward", timing_raw, color="yellow"):
                batch = self._compute_reward_colocate(batch)

        if self.config.algorithm.adv_estimator == core_algos.AdvantageEstimator.REMAX:
            batch = self._add_remax_reward_baselines(batch)

        # 3.5 [OPTIONAL] build SDPO teacher reprompts and distillation masks. Runs before balancing
        # so the synthetic padding samples inherit the teacher fields from the template sample.
        self._maybe_build_self_distillation_batch(batch, metrics=metrics)

        # 4. balance batch across data parallel groups
        batch = self._balance_batch(batch, metrics=metrics)

        # 5. compute old_log_prob
        with marked_timer("old_log_prob", timing_raw, color="blue"):
            batch = self._compute_old_log_prob(batch, metrics=metrics)

        # 6. [OPTIONAL] compute ref_log_prob
        if self.use_reference_policy:
            with marked_timer("ref", timing_raw, color="olive"):
                batch = self._compute_ref_log_prob(batch, metrics=metrics)

        # 7. [OPTIONAL] compute critic values
        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                batch = self._compute_values(batch, metrics=metrics)

        # 8. compute advantage and return
        with marked_timer("adv", timing_raw, color="brown"):
            batch = self._compute_advantage(batch, metrics=metrics)

        # 9. [OPTIONAL] update critic
        if self.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                batch = self._update_critic(batch, metrics=metrics)

        # 10. update actor
        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer("update_actor", timing_raw, color="red"):
                batch = self._update_actor(batch, metrics=metrics)

        return batch


@ray.remote
class TaskRunner:
    def __init__(self) -> None:
        # role => worker class
        self.role_worker_mapping = {}
        # role => resource pool
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker to mapping."""
        # SDPO validation
        self_distillation_cfg = config.actor_rollout_ref.actor.get("self_distillation", None)
        loss_mode = config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        self_distillation_needs_ref = self_distillation_cfg is not None and loss_mode == "sdpo"
        if self_distillation_needs_ref and need_reference_policy(config):
            raise ValueError("SDPO cannot share the reference policy with KL regularization.")
        if self_distillation_needs_ref and config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
            raise ValueError("SDPO currently supports FSDP/FSDP2 actor strategy only.")

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        # Ref policy is fused into ActorRolloutRefWorker unless LoRA is used with a dedicated ref model.
        # For SDPO, always use ActorRolloutRef so teacher inference has both ref and actor modules.
        if (need_reference_policy(config) and not ref_in_actor) or self_distillation_needs_ref:
            role = Role.ActorRolloutRef
        else:
            role = Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(ActorRolloutRefWorker)
        self.mapping[role] = "global_pool"

    def add_critic_worker(self, config):
        """Add critic worker to mapping."""
        if need_critic(config):
            self.role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)
            self.mapping[Role.Critic] = "global_pool"

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""

        # Global resource pool is used for actor, rollout, critic, ref
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        # Add separate resource pool for reward model if enabled
        if config.reward.reward_model.enable_resource_pool:
            if config.reward.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward.reward_model.nnodes <= 0:
                raise ValueError("config.reward.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward.reward_model.n_gpus_per_node] * config.reward.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool
            self.mapping[Role.RewardModel] = "reward_pool"
        else:
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node
            self.mapping[Role.RewardModel] = "global_pool"

        distillation_config = config.get("distillation")
        if is_distillation_enabled(distillation_config):
            if distillation_config.n_gpus_per_node <= 0:
                raise ValueError("config.distillation.n_gpus_per_node must be greater than 0")
            if distillation_config.nnodes <= 0:
                raise ValueError("config.distillation.nnodes must be greater than 0")

            teacher_pool = [distillation_config.n_gpus_per_node] * distillation_config.nnodes
            resource_pool_spec["teacher_pool"] = teacher_pool
            self.mapping[Role.TeacherModel] = "teacher_pool"

        self.resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def run(self, config):
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        from verl.utils.debug_breakpoints import should_break
        if should_break("taskrunner"): breakpoint()

        # initialize transfer queue
        tq.init(config.transfer_queue)
        trainer = None
        try:
            self.add_actor_rollout_worker(config)
            self.add_critic_worker(config)
            self.init_resource_pool_mgr(config)

            trainer = PPOTrainer(
                config=config,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=self.resource_pool_manager,
            )
            trainer.init_workers()
            trainer.fit()
        finally:
            if trainer:
                trainer.replay_buffer.close()
            tq.close()


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)

    config.transfer_queue.enable = True

    # validate config
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )

    run_ppo(config, task_runner_class=TaskRunner)


if __name__ == "__main__":
    main()
