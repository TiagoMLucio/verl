# Deferred upstream patches

## pr6660-qwen35-gdn-packed-cu-seqlens.patch
Upstream verl #6660: pass packed `cu_seqlens` into Qwen3.5's GDN linear-attention
layers under remove-padding / Ulysses SP. Without it, rmpad silently leaks
linear-attention state across packed sequence boundaries (corrupt logprobs).

NOT applied yet: its `transformer_impl.py` hunks overlap the fork's SDPO span-only
lm_head changes and need a careful hand-merge. Until then Qwen3.5 runs MUST set
`actor_rollout_ref.model.use_remove_padding=False` (enforced by
cluster/clariden/train.sbatch's QWEN35_OVERRIDES). Merge this patch when
re-enabling rmpad + flash-attn for throughput.
