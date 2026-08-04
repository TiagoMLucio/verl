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
"""Top-k log-probabilities over a linear head, chunked across positions."""

from typing import Optional

import torch


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Half precision is accumulated in fp32; anything wider is left alone, so a float64
    caller (gradcheck) is not silently truncated."""
    return torch.float32 if dtype in (torch.bfloat16, torch.float16) else dtype


def _chunk_logits(hidden: torch.Tensor, weight: torch.Tensor, temperature: Optional[torch.Tensor]) -> torch.Tensor:
    logits = (hidden @ weight.t()).to(_compute_dtype(hidden.dtype))
    if temperature is not None:
        logits = logits / temperature.to(logits.dtype)
    return logits


class ChunkedTopkLogprobs(torch.autograd.Function):
    """log_softmax(logits).topk(k) without ever materializing (positions, vocab).

    Each chunk of positions is projected, reduced to its top-k and logsumexp, then freed;
    the backward projects it again from the saved hidden states. Peak memory is set by
    ``chunk_size`` rather than by the number of scored positions, which is what lets the
    number of hinted spans grow without the update growing with it.
    """

    @staticmethod
    def forward(ctx, hidden, weight, k, temperature=None, chunk_size=512):
        num_positions = hidden.shape[0]
        compute_dtype = _compute_dtype(hidden.dtype)
        topk_logps = hidden.new_empty((num_positions, k), dtype=compute_dtype)
        topk_indices = hidden.new_empty((num_positions, k), dtype=torch.int64)

        for start in range(0, num_positions, chunk_size):
            end = min(start + chunk_size, num_positions)
            temp = None if temperature is None else temperature[start:end]
            logits = _chunk_logits(hidden[start:end], weight, temp)
            values, indices = logits.topk(k, dim=-1)
            topk_logps[start:end] = values - logits.logsumexp(dim=-1, keepdim=True)
            topk_indices[start:end] = indices

        ctx.save_for_backward(hidden, weight, topk_indices, temperature)
        ctx.chunk_size = chunk_size
        ctx.mark_non_differentiable(topk_indices)
        return topk_logps, topk_indices

    @staticmethod
    def backward(ctx, grad_topk_logps, _grad_indices):
        hidden, weight, topk_indices, temperature = ctx.saved_tensors
        chunk_size = ctx.chunk_size

        compute_dtype = _compute_dtype(hidden.dtype)
        grad_hidden = torch.zeros_like(hidden) if ctx.needs_input_grad[0] else None
        grad_weight = torch.zeros_like(weight, dtype=compute_dtype) if ctx.needs_input_grad[1] else None
        grad_topk_logps = grad_topk_logps.to(compute_dtype)

        for start in range(0, hidden.shape[0], chunk_size):
            end = min(start + chunk_size, hidden.shape[0])
            temp = None if temperature is None else temperature[start:end]
            logits = _chunk_logits(hidden[start:end], weight, temp)

            # d(topk_logits - logsumexp)/d(logits) = onehot(topk_indices) - softmax(logits)
            grad_chunk = torch.zeros_like(logits)
            grad_chunk.scatter_(-1, topk_indices[start:end], grad_topk_logps[start:end])
            grad_chunk -= logits.softmax(dim=-1) * grad_topk_logps[start:end].sum(dim=-1, keepdim=True)
            if temp is not None:
                grad_chunk = grad_chunk / temp

            if grad_hidden is not None:
                grad_hidden[start:end] = (grad_chunk @ weight.to(compute_dtype)).to(hidden.dtype)
            if grad_weight is not None:
                grad_weight += grad_chunk.t() @ hidden[start:end].to(compute_dtype)

        if grad_weight is not None:
            grad_weight = grad_weight.to(weight.dtype)
        return grad_hidden, grad_weight, None, None, None


def chunked_topk_logprobs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    k: int,
    temperature: Optional[torch.Tensor] = None,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k log-probabilities and their vocabulary indices for each position in ``hidden``.

    Args:
        hidden: (num_positions, hidden_size) states of the positions being scored.
        weight: (vocab_size, hidden_size) output embedding.
        k: how many vocabulary entries to keep per position.
        temperature: optional (num_positions, 1) per-position temperature.
        chunk_size: positions projected at once; sets the peak, not the result.

    Returns:
        (num_positions, k) log-probabilities and their (num_positions, k) indices.
    """
    return ChunkedTopkLogprobs.apply(hidden, weight, k, temperature, chunk_size)


def chunked_gather_logprobs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    indices: torch.Tensor,
    temperature: Optional[torch.Tensor] = None,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Log-probabilities at ``indices``, chunked the same way. For the teacher, which scores
    the student's chosen entries and never needs a gradient."""
    num_positions = hidden.shape[0]
    out = hidden.new_empty((num_positions, indices.shape[-1]), dtype=_compute_dtype(hidden.dtype))
    for start in range(0, num_positions, chunk_size):
        end = min(start + chunk_size, num_positions)
        temp = None if temperature is None else temperature[start:end]
        logits = _chunk_logits(hidden[start:end], weight, temp)
        gathered = torch.gather(logits, -1, indices[start:end])
        out[start:end] = gathered - logits.logsumexp(dim=-1, keepdim=True)
    return out
