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
"""The sync trainer's check of an SDPO run's KL settings, before any worker starts."""

import pytest
from omegaconf import OmegaConf

from verl.trainer.main_ppo_sync import check_sdpo_kl


def sdpo_config(use_kl_in_reward=False, use_fused_kernels=True, use_chunked_lm_head=False, **actor):
    actor = {
        "use_kl_loss": True,
        "loss_agg_mode": "traj-mean-token-mean",
        "self_distillation": {"teacher_regularization": "ema"},
        **actor,
    }
    model = {"use_fused_kernels": use_fused_kernels, "use_chunked_lm_head": use_chunked_lm_head}
    return OmegaConf.create(
        {"algorithm": {"use_kl_in_reward": use_kl_in_reward}, "actor_rollout_ref": {"actor": actor, "model": model}}
    )


def test_an_sdpo_run_takes_the_kl_loss_on_its_ema_teacher():
    check_sdpo_kl(sdpo_config())
    check_sdpo_kl(sdpo_config(self_distillation={"teacher_regularization": "none"}))


def test_without_the_kl_loss_nothing_else_is_asked_of_the_run():
    check_sdpo_kl(sdpo_config(use_kl_loss=False, loss_agg_mode="token-mean", use_fused_kernels=False))


@pytest.mark.parametrize(
    "overrides, match",
    [
        pytest.param({"use_kl_in_reward": True, "use_kl_loss": False}, "advantages", id="kl in reward"),
        pytest.param(
            {"self_distillation": {"teacher_regularization": "trust_region"}}, "trust-region", id="trust region"
        ),
        pytest.param({"loss_agg_mode": "token-mean"}, "traj-mean-token-mean", id="token-mean"),
        pytest.param({"use_fused_kernels": False}, "fused kernel", id="no fused kernel"),
        pytest.param({"use_chunked_lm_head": True}, "fused kernel", id="chunked head"),
    ],
)
def test_a_kl_setting_the_run_cannot_honour_is_refused(overrides, match):
    with pytest.raises(ValueError, match=match):
        check_sdpo_kl(sdpo_config(**overrides))
