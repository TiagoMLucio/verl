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


from typing import Any, Optional

import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_self_distillation_loss,
    compute_value_loss,
    get_policy_loss_fn,
    kl_penalty,
)
from verl.utils import tensordict_utils as tu
from verl.utils.attention_utils import index_first_axis, rearrange
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.device import get_device_name
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]
    if not batch_num_tokens:
        # zero-supervised mini: 0/0 would NaN the loss value (its grads are already zero)
        batch_num_tokens = 1

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    # select fields and convert to padded tensor
    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics


def _sdpo_response_topk_indices_to_no_padding(
    data: TensorDict, logits: torch.Tensor, logits_keep_idx: Optional[torch.Tensor] = None
) -> Optional[torch.Tensor]:
    """Map padded response top-k indices onto the flattened no-padding sequence."""
    topk_indices = tu.get(data, "student_topk_indices", default=None)
    if topk_indices is None:
        return None

    response_length = data["responses"].shape[-1]
    topk = topk_indices.size(-1)
    batch_size = topk_indices.shape[0]
    max_seq_len = tu.get_non_tensor_data(data=data, key="max_seq_len", default=None)

    full_topk_indices = torch.zeros(
        batch_size,
        max_seq_len,
        topk,
        device=topk_indices.device,
        dtype=topk_indices.dtype,
    )
    full_topk_indices[:, -response_length - 1 : -1, :] = topk_indices

    indices = tu.get_non_tensor_data(data=data, key="indices", default=None)
    gathered_indices = index_first_axis(rearrange(full_topk_indices, "b s k -> (b s) k"), indices)
    if logits_keep_idx is not None:
        gathered_indices = gathered_indices[logits_keep_idx]
    sp_size = tu.get_non_tensor_data(data=data, key="sp_size", default=1)
    if sp_size > 1:
        from verl.utils.ulysses import slice_input_tensor

        gathered_indices = slice_input_tensor(gathered_indices.unsqueeze(0), dim=1, padding=True).squeeze(0)
    return gathered_indices[: logits.shape[0]]


def _sdpo_logits_processor(student_logits: torch.Tensor, sdpo_config) -> dict:
    """Logits-processor call during actor forward: compute student top-k or full logps.

    Keeps the (1, total_nnz, ...) leading batch dim so the engine's squeeze(0) stays
    unambiguous when only a single packed position is kept."""
    logits = student_logits  # (1, total_nnz, vocab_size)
    distill_topk = sdpo_config.distillation_topk
    if distill_topk is not None:
        topk = min(distill_topk, logits.shape[-1])
        topk_logits, topk_indices = torch.topk(logits, topk, dim=-1)
        logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        topk_logps = topk_logits - logsumexp
        return {"topk_logps": topk_logps, "topk_indices": topk_indices}
    all_logps = torch.log_softmax(logits, dim=-1)
    return {"all_logps": all_logps}


def _sdpo_teacher_extractor(
    student_logits: Optional[torch.Tensor] = None,
    data=None,
    logits_keep_idx: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict | tuple[torch.Tensor, dict]:
    """Teacher logits processor for full-vocab or student-top-k SDPO targets.

    With ``logits_keep_idx``, ``student_logits`` holds only those packed positions
    (span-only lm_head) and the top-k index map is subset accordingly.
    """
    if student_logits is None:
        return torch.tensor(1.0, device=get_device_name()), {}
    logits = student_logits.squeeze(0)
    topk_indices = _sdpo_response_topk_indices_to_no_padding(
        data=data, logits=logits, logits_keep_idx=logits_keep_idx
    )
    # outputs keep the (1, total_nnz, ...) batch dim: engine contract shared with _sdpo_logits_processor
    if topk_indices is None:
        return {"all_logps": torch.log_softmax(logits, dim=-1).unsqueeze(0)}
    topk_logits = torch.gather(logits, dim=-1, index=topk_indices)
    logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
    return {"topk_logps": (topk_logits - logsumexp).unsqueeze(0)}


def sdpo_ppo_loss(
    config: ActorConfig,
    sdpo_config,
    teacher_logprob_fn=None,
    student_logits: Optional[torch.Tensor] = None,
    model_output: Optional[dict] = None,
    data: Optional[TensorDict] = None,
    dp_group=None,
    logits_keep_idx: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]] | dict:
    """SDPO loss function used as both logits processor and final policy loss."""
    if student_logits is not None:
        # span-only passes already subset student_logits; logits_keep_idx is only needed
        # by the teacher extractor's top-k index map
        return _sdpo_logits_processor(student_logits=student_logits, sdpo_config=sdpo_config)

    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = None

    response_mask = tu.get(data, "response_mask", default=None)
    if response_mask is None:
        raise ValueError("SDPO: response_mask missing in data.")

    # Convert the response-shaped fields from the no-padding (nested) layout to the padded
    # layout the distillation loss expects; this is a no-op for already-padded tensors
    # (e.g. the legacy worker path). Mirrors the ppo_loss convention above. The original
    # ``data`` is left untouched because it is still consumed below (student no_padding_2_padding
    # calls and the teacher_logprob_fn).
    pad_fields = [
        field
        for field in ("response_mask", "self_distillation_mask", "old_log_probs", "rollout_is_weights")
        if field in data
    ]
    # one row per condensation segment: each row weighs its share of the trajectory's supervised tokens
    trace_weight = tu.get(data, "trace_weight", default=None)
    if trace_weight is not None:
        if trace_weight.is_nested:
            trace_weight = trace_weight.to_padded_tensor(0.0)
        trace_weight = trace_weight.reshape(-1).to(torch.float32)
    padded = data.select(*pad_fields).to_padded_tensor()
    response_mask = padded["response_mask"]
    self_distillation_mask = tu.get(padded, "self_distillation_mask", default=None)
    old_log_probs = tu.get(padded, "old_log_probs", default=None)
    rollout_is_weights = tu.get(padded, "rollout_is_weights", default=None)

    # collapse the TQ-materialized (batch, 1) scalar mask; per-token (batch, response_len) masks pass through
    if self_distillation_mask is not None and self_distillation_mask.dim() > 1 and self_distillation_mask.shape[1] == 1:
        self_distillation_mask = self_distillation_mask.reshape(self_distillation_mask.shape[0])

    full_logit_distillation = sdpo_config.full_logit_distillation
    distill_topk = sdpo_config.distillation_topk
    student_topk_logps = None
    student_topk_indices = None
    student_all_logps = None
    teacher_topk_logps = None
    teacher_all_logps = None

    if full_logit_distillation and distill_topk is not None:
        student_topk_logps = no_padding_2_padding(model_output["topk_logps"], data)
        student_topk_indices = no_padding_2_padding(model_output["topk_indices"], data)
    elif full_logit_distillation:
        student_all_logps = no_padding_2_padding(model_output["all_logps"], data)

    if teacher_logprob_fn is not None:
        teacher_outputs = teacher_logprob_fn(
            data=data,
            student_topk_indices=student_topk_indices,
            return_all_logps=full_logit_distillation and distill_topk is None,
        )
        teacher_log_probs = teacher_outputs.get("teacher_log_probs", teacher_log_probs)
        teacher_topk_logps = teacher_outputs.get("teacher_topk_log_probs", teacher_topk_logps)
        teacher_all_logps = teacher_outputs.get("teacher_all_log_probs", teacher_all_logps)

    if teacher_log_probs is None:
        raise ValueError("SDPO: teacher_log_probs missing and no teacher_logprob_fn produced it.")
    if full_logit_distillation and distill_topk is not None and teacher_topk_logps is None:
        raise ValueError("SDPO: teacher_topk_log_probs missing for full-logit top-k distillation.")
    if full_logit_distillation and distill_topk is None and teacher_all_logps is None:
        raise ValueError("SDPO: teacher_all_log_probs missing for full-logit distillation.")

    # batch_num_tokens is the global supervised-token count: the SDPO update ships loss_mask = supervised mask
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    loss, metrics = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=sdpo_config,
        old_log_probs=old_log_probs,
        student_all_log_probs=student_all_logps,
        teacher_all_log_probs=teacher_all_logps,
        student_topk_log_probs=student_topk_logps,
        teacher_topk_log_probs=teacher_topk_logps,
        self_distillation_mask=self_distillation_mask,
        loss_agg_mode=config.loss_agg_mode,
        rollout_is_weights=rollout_is_weights,
        global_batch_info=config.global_batch_info,
        seq_weights=trace_weight,
    )
    metrics["self_distillation/empty_target_batch"] = (
        self_distillation_mask.sum().item() == 0 if self_distillation_mask is not None else False
    )

    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    # '__sum' pairs must add across micro-batches; everything else is a per-micro-batch mean
    summed = {k: v for k, v in metrics.items() if k.endswith("__sum")}
    metrics = Metric.from_dict(
        {k: v for k, v in metrics.items() if not k.endswith("__sum")}, aggregation=AggregationType.MEAN
    )
    metrics.update(Metric.from_dict(summed, aggregation=AggregationType.SUM))
    metrics["actor/pg_loss"] = Metric(value=loss, aggregation=metric_aggregation)
    policy_loss = loss

    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)
        # span-only update: entropy exists only at hinted-span positions, so average there
        entropy_mask = response_mask
        if "logits_keep_positions" in data.keys() and self_distillation_mask is not None and self_distillation_mask.dim() > 1:
            entropy_mask = response_mask * self_distillation_mask
        # a 0 batch_num_tokens would make entropy_loss inf with NaN grads (entropy connects to the graph unmasked)
        entropy_gbi = dict(config.global_batch_info or {})
        if not entropy_gbi.get("batch_num_tokens"):
            entropy_gbi["batch_num_tokens"] = entropy_mask.sum().clamp(min=1.0)
        entropy_loss = agg_loss(
            loss_mat=entropy,
            loss_mask=entropy_mask,
            loss_agg_mode=config.loss_agg_mode,
            **entropy_gbi,
        )
        policy_loss -= config.entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)
        # Is the policy sharpening only where we train it, or everywhere? Summed, not meaned:
        # un-hinted micro-batches would otherwise drag the average toward zero.
        metrics["actor/entropy_hinted__sum"] = Metric(
            value=(entropy.detach() * entropy_mask).sum(), aggregation=AggregationType.SUM
        )
        metrics["actor/entropy_hinted_tokens__sum"] = Metric(
            value=entropy_mask.sum(), aggregation=AggregationType.SUM
        )

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)

    # select fields and convert to padded tensor
    data = data.select("values", "returns", "response_mask").to_padded_tensor()
    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics
