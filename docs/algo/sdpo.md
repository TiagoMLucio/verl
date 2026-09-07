# Self-Distillation Policy Optimization (SDPO)

Last updated: 09/04/2026.

SDPO is a policy optimization variant that augments actor updates with self-distillation from successful trajectories in the same rollout batch.

Paper: [Self-Distillation Policy Optimization](https://arxiv.org/abs/2601.20802).

## Core Idea

At each training step:

1. Rollout responses are grouped by sample `uid`.
2. Successful responses (above `success_reward_threshold`) are reused as demonstrations.
3. Optionally, environment feedback is included in the reprompt.
4. A teacher prompt is built per sample and concatenated with the original response tokens.
5. The actor is updated with PPO-style optimization where policy loss is replaced by SDPO distillation loss (`loss_mode=sdpo`).
6. A colocated teacher is updated with EMA from the student weights.

The teacher that builds steps 2-4 is the `self_distillation.teacher` block: a `_target_` naming an `SDPOTeacher` subclass (`verl.trainer.ppo.sdpo.SDPOTeacher`) plus that teacher's own options. The config group `verl/trainer/config/sdpo_teacher/` holds one file per teacher:

- `reprompt.yaml` (`verl.trainer.ppo.sdpo.RepromptTeacher`, the trainer default): the paper's sibling solution plus feedback, with `max_reprompt_len`, `reprompt_truncation`, `dont_reprompt_on_self_success`, `remove_thinking_from_demonstration`, the three templates and `environment_feedback_only_without_solution`.
- `turn_hints.yaml` (`uni_agent.sdpo.TurnHintTeacher`, project code outside verl): each reflector hint spliced into the trajectory right before the turn it was written for, only that turn's span scored, with `turn_hint_template`, `call_hint_template`, `chat_template_kwargs`, `max_hinted_turns` and `call_loss_weight`.

The trainer builds the teacher with `make_teacher` (`hydra.utils.instantiate` on the block, handing it the tokenizer, the prompt budget, the dataset's `apply_chat_template_kwargs` and `success_reward_threshold`); `_recursive_: false` on the `self_distillation` block keeps hydra from building it with the actor config. A teacher's options are keyword-only constructor parameters, so an unknown key fails at construction.

## Key Configs

- `actor_rollout_ref.actor.policy_loss.loss_mode: sdpo`
- `actor_rollout_ref.actor.self_distillation.full_logit_distillation`
- `actor_rollout_ref.actor.self_distillation.distillation_topk`
- `actor_rollout_ref.actor.self_distillation.alpha`
- `actor_rollout_ref.actor.self_distillation.success_reward_threshold`
- `sdpo_teacher@actor_rollout_ref.actor.self_distillation.teacher` (config group: `reprompt`, the paper's sibling solution plus feedback, or `turn_hints`, per-turn reflector hints spliced into the trajectory)
- `actor_rollout_ref.actor.self_distillation.teacher_regularization` (`ema`, `trust_region`, or `none`)
- `actor_rollout_ref.actor.self_distillation.teacher_update_rate`
- `actor_rollout_ref.actor.self_distillation.include_environment_feedback`
- `actor_rollout_ref.actor.self_distillation.teacher.<option>` (the selected teacher's own keys, e.g. `teacher.environment_feedback_only_without_solution` under `reprompt`)

## Current Constraints in verl

- SDPO requires `fsdp` / `fsdp2` actor strategy.
- SDPO cannot be combined with a separate KL reference policy (`use_kl_in_reward` or `use_kl_loss` reference path).
- Distillation with multimodal actor inputs is currently not supported.
- Trust-region teacher regularization requires `actor_rollout_ref.actor.use_fused_kernels=false`.

## Minimal Usage

Use the preset config:

```bash
python3 -m verl.trainer.main_ppo --config-name sdpo
```

Or override from `ppo_trainer`:

```bash
python3 -m verl.trainer.main_ppo \
  actor_rollout_ref.actor.policy_loss.loss_mode=sdpo \
  actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.05
```

Select the turn-hint teacher and set one of its options:

```bash
python3 -m verl.trainer.main_ppo_sync --config-name sdpo \
  sdpo_teacher@actor_rollout_ref.actor.self_distillation.teacher=turn_hints \
  actor_rollout_ref.actor.self_distillation.teacher.max_hinted_turns=2
```
