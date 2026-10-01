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
"""The naive weight sync wakes a rank's kv_cache only after every rank has offloaded its params."""

import asyncio
import time
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import torch.distributed as dist
import torch.multiprocessing as mp

from verl.workers import engine_workers
from verl.workers.engine_workers import ActorRolloutRefWorker


def _worker(rank, out_dir):
    def offload(*args, **kwargs):
        if rank == 1:
            time.sleep(1.0)
        (out_dir / f"offloaded_{rank}").touch()

    async def resume(tags):
        if tags == ["kv_cache"]:
            (out_dir / f"peer_offloaded_at_kv_resume_{rank}.txt").write_text(
                repr((out_dir / f"offloaded_{1 - rank}").exists())
            )

    worker = object.__new__(ActorRolloutRefWorker)
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(free_cache_engine=True, checkpoint_engine=SimpleNamespace(backend="naive"))
    )
    worker.actor = SimpleNamespace(
        engine=MagicMock(
            get_per_tensor_param=MagicMock(return_value=(iter(()), None)),
            is_param_offload_enabled=True,
            to=MagicMock(side_effect=offload),
        )
    )
    worker.rollout = AsyncMock(resume=AsyncMock(side_effect=resume))
    worker.layered_summon = False
    worker.peft_merge = False
    worker.base_sync_done = True
    return worker


def _rank(rank, world, store, out_dir):
    engine_workers.set_expandable_segments = lambda enable: None
    engine_workers.log_gpu_memory_usage = lambda *args, **kwargs: None
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", world_size=world, rank=rank, timeout=timedelta(seconds=30)
    )
    try:
        asyncio.run(_worker(rank, out_dir).update_weights())
    finally:
        dist.destroy_process_group()


def test_kv_cache_resumes_after_every_rank_offloaded(tmp_path):
    """Rank 1 offloads a second late; rank 0 must not wake its kv_cache before then."""
    mp.spawn(_rank, args=(2, tmp_path / "store", tmp_path), nprocs=2, join=True)
    seen = [(tmp_path / f"peer_offloaded_at_kv_resume_{rank}.txt").read_text() for rank in range(2)]
    assert seen == ["True", "True"]
