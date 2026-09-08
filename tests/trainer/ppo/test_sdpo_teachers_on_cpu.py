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
"""The reprompt teacher on a toy tokenizer: it builds the paper's messages, masks whole rows
and decodes only the responses it uses as a solution; ``self_distillation.teacher`` names it
by ``_target_`` and owns its options; and the trainer's teacher-build step writes the five
fields and the batch metrics a reprompt run logs."""

import inspect
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

import verl
from verl.trainer import main_ppo_sync
from verl.trainer.ppo.sdpo import RepromptTeacher, SDPOTeacher, TeacherInputs, make_teacher
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config.actor import SelfDistillationConfig

CONFIG_DIR = Path(verl.__file__).parent / "trainer" / "config"
REPROMPT_TARGET = "verl.trainer.ppo.sdpo.RepromptTeacher"


class ToyTokenizer:
    """Character tokens (id = ord), a chat template that renders ``<role>content</>`` per turn
    and ``<assistant>`` as the generation header; batch tokenization honours padding_side and
    truncation_side the way the HF one does."""

    eos_token = "<eos>"
    pad_token_id = 0

    def __init__(self):
        self.padding_side = "right"
        self.truncation_side = "right"
        self.decode_calls = 0
        self.last_batch = None

    @staticmethod
    def render(messages, add_generation_prompt):
        text = "".join(f"<{m['role']}>{m['content']}</>" for m in messages)
        return text + "<assistant>" if add_generation_prompt else text

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        self.decode_calls += 1
        return "".join(chr(int(i)) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        if not tokenize:
            return self.render(messages, add_generation_prompt)
        self.last_batch = messages
        max_length = kwargs["max_length"]
        rows = [self.encode(self.render(conv, add_generation_prompt)) for conv in messages]
        rows = [r[-max_length:] if self.truncation_side == "left" else r[:max_length] for r in rows]
        width = max(len(r) for r in rows)
        ids = torch.zeros((len(rows), width), dtype=torch.int64)
        mask = torch.zeros((len(rows), width), dtype=torch.int64)
        for i, r in enumerate(rows):
            sl = slice(width - len(r), width) if self.padding_side == "left" else slice(0, len(r))
            ids[i, sl] = torch.tensor(r)
            mask[i, sl] = 1
        return {"input_ids": ids, "attention_mask": mask}


def ids(text):
    return torch.tensor([ord(c) for c in text], dtype=torch.int64)


HEADER = "<assistant>"
PROMPT = ids("<user>task</>" + HEADER)
# turn 0, an observation, turn 1 (reasoning then a tool call)
TURN0, OBS, TURN1 = "abc</>", "<user>obs</>" + HEADER, "def<tool_call>x</>"
RESPONSE = ids(TURN0 + OBS + TURN1)
SPANS = [[0, 0, len(TURN0)], [1, len(TURN0 + OBS), len(TURN0 + OBS + TURN1)]]
SEGMENT_PROMPT = [{"role": "user", "content": "condensed"}, {"role": "assistant", "content": "so far"}]


def _inputs(extra_fields, uids, seq_scores, feedback, responses=None):
    n = len(extra_fields)
    responses = responses or [RESPONSE.clone() for _ in range(n)]
    mask = [torch.ones(r.shape[0], dtype=torch.int64) for r in responses]
    for m in mask:
        m[1] = 0  # a tool-observation token the student never wrote
    return TeacherInputs(
        prompts=[PROMPT.clone() for _ in range(n)],
        responses=responses,
        response_mask=mask,
        raw_prompts=[
            [{"role": "system", "content": "sys"}, {"role": "user", "content": f"task {i}"}] for i in range(n)
        ],
        uids=list(uids),
        seq_scores=list(seq_scores),
        feedback=list(feedback),
        extra_fields=extra_fields,
        traj_of_row=[f"{uid}_{i}" for i, uid in enumerate(uids)],
    )


def _sd_config(**teacher_options):
    return SelfDistillationConfig(teacher={"_target_": REPROMPT_TARGET, **teacher_options})


def test_reprompt_teacher_messages_masks_and_lazy_decode():
    tok = ToyTokenizer()
    cfg = SelfDistillationConfig(
        include_environment_feedback=True,
        teacher=dict(
            _target_=REPROMPT_TARGET,
            dont_reprompt_on_self_success=True,
            remove_thinking_from_demonstration=True,
            environment_feedback_only_without_solution=False,
            max_reprompt_len=512,
            reprompt_truncation="left",
        ),
    )
    teacher = make_teacher(cfg, tok, max_prefix_len=4096, apply_chat_template_kwargs={})
    assert isinstance(teacher, RepromptTeacher) and not teacher.needs_prompts
    assert teacher.success_reward_threshold == cfg.success_reward_threshold
    assert (teacher.max_reprompt_len, teacher.reprompt_truncation) == (512, "left")
    # uid a: row 0 failed with feedback, row 1 solved (its solution serves row 0, not itself);
    # uid b: row 2 failed without feedback, row 3 a condensation segment with feedback
    extra = [{}, {}, {}, {"segment_index": 1, "segment_prompt": SEGMENT_PROMPT}]
    responses = [RESPONSE.clone(), ids("<think>t</think>sol"), RESPONSE.clone(), RESPONSE.clone()]
    inputs = _inputs(extra, ["a", "a", "b", "b"], [0.0, 1.0, 0.0, 0.0], ["fb0", None, None, "fb3"], responses)

    out = teacher.build(inputs)

    assert tok.decode_calls == 1, "only the response used as a solution is decoded"
    assert (tok.padding_side, tok.truncation_side) == ("right", "right"), "tokenizer sides are restored"
    assert set(out.fields) == {"teacher_input_ids", "self_distillation_mask", "loss_mask"}
    assert out.metrics == {}

    solution = teacher.solution_template.format(successful_previous_attempt="sol")
    feedback0 = teacher.feedback_template.format(feedback_raw="fb0")
    feedback3 = teacher.feedback_template.format(feedback_raw="fb3")
    reprompt0 = teacher.reprompt_template.format(prompt="task 0", solution=solution, feedback=feedback0)
    reprompt3 = teacher.reprompt_template.format(prompt="", solution="", feedback=feedback3)
    expected = [
        [{"role": "system", "content": "sys"}, {"role": "user", "content": reprompt0}],
        inputs.raw_prompts[1],
        inputs.raw_prompts[2],
        SEGMENT_PROMPT + [{"role": "user", "content": reprompt3}],
    ]
    assert tok.last_batch == expected

    assert out.fields["self_distillation_mask"].tolist() == [1.0, 0.0, 0.0, 1.0]
    for i, conv in enumerate(expected):
        prompt = ids(ToyTokenizer.render(conv, add_generation_prompt=True))
        assert torch.equal(out.fields["teacher_input_ids"][i], torch.cat([prompt, responses[i]]))
        used = int(out.fields["self_distillation_mask"][i])
        assert torch.equal(out.fields["loss_mask"][i], inputs.response_mask[i] * used)


def test_reprompt_truncation_side_scoped_to_the_reprompt():
    tok = ToyTokenizer()
    inputs = _inputs([{}], ["a"], [1.0], [None])
    teacher = RepromptTeacher(tok, success_reward_threshold=1.0, max_reprompt_len=8, reprompt_truncation="left")
    out = teacher.build(inputs)
    prompt = ids(ToyTokenizer.render(inputs.raw_prompts[0], add_generation_prompt=True))
    assert torch.equal(out.fields["teacher_input_ids"][0][:8], prompt[-8:]), "left-truncated to max_reprompt_len"
    assert tok.truncation_side == "right"


def _qwen35_tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    except OSError:
        pytest.skip("Qwen/Qwen3.5-4B tokenizer is not in the local HF cache")


def _generation_header(tok, kwargs):
    """The tokens apply_chat_template appends for the assistant turn about to be sampled."""
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    without = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, **kwargs)
    with_header = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    assert with_header.startswith(without)
    return tok.encode(with_header[len(without) :], add_special_tokens=False)


def test_reprompt_header_follows_chat_template_kwargs():
    """The response was sampled after the rollout's generation header, so the reprompt has to
    end with the same one: under Qwen3.5 that is 7 tokens with enable_thinking off and 5 with
    the template's default, which is what the dataset's empty kwargs render."""
    tok = _qwen35_tokenizer()
    rollout_header = _generation_header(tok, {"enable_thinking": False})
    default_header = _generation_header(tok, {})
    assert len(rollout_header) == 7 and len(default_header) == 5
    assert tok.decode(rollout_header) == "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    inputs = _inputs([{}], ["a"], [0.0], ["fb"], responses=[torch.tensor([9, 9, 9])])
    for kwargs, header in (({"enable_thinking": False}, rollout_header), ({}, default_header)):
        teacher = RepromptTeacher(tok, success_reward_threshold=0.5, chat_template_kwargs=kwargs)
        prompt = teacher.build(inputs).fields["teacher_input_ids"][0][:-3]
        assert prompt[-len(header) :].tolist() == header, kwargs
        assert tok.decode(prompt).endswith(tok.decode(header))


def test_teacher_options_are_validated_at_construction():
    tok = ToyTokenizer()
    with pytest.raises(TypeError, match="max_hinted_turns"):
        make_teacher(_sd_config(max_hinted_turns=1), tok, max_prefix_len=4096)
    with pytest.raises(ValueError, match="reprompt_truncation"):
        make_teacher(_sd_config(reprompt_truncation="middle"), tok, max_prefix_len=4096)
    with pytest.raises(ValueError, match="_target_"):
        SelfDistillationConfig(teacher={"max_reprompt_len": 8})
    with pytest.raises(ValueError, match="_target_"):
        make_teacher(SimpleNamespace(teacher={}, success_reward_threshold=1.0), tok, max_prefix_len=4096)
    with pytest.raises(TypeError, match="SDPOTeacher"):
        make_teacher(
            SimpleNamespace(teacher={"_target_": "builtins.dict"}, success_reward_threshold=1.0),
            tok,
            max_prefix_len=4096,
        )
    assert issubclass(RepromptTeacher, SDPOTeacher) and not SDPOTeacher.needs_prompts


def test_reprompt_yaml_is_the_teacher_default_and_hydra_leaves_the_block_alone():
    """The config group file is the paper's teacher: its keys are the constructor's keyword
    options at the same values, and the actor-level hydra instantiation (the worker's
    ``omega_conf_to_dataclass``) hands the block over untouched for ``make_teacher``."""
    block = yaml.safe_load((CONFIG_DIR / "actor" / "actor.yaml").read_text())["self_distillation"]
    block["teacher"] = yaml.safe_load((CONFIG_DIR / "sdpo_teacher" / "reprompt.yaml").read_text())
    assert block["_recursive_"] is False

    cfg = omega_conf_to_dataclass(OmegaConf.create(block))
    assert isinstance(cfg, SelfDistillationConfig)
    assert cfg.teacher["_target_"] == REPROMPT_TARGET
    structured = omega_conf_to_dataclass(OmegaConf.create(block), SelfDistillationConfig)
    assert structured.teacher == cfg.teacher

    teacher = make_teacher(cfg, ToyTokenizer(), max_prefix_len=4096, apply_chat_template_kwargs={"a": 1})
    assert isinstance(teacher, RepromptTeacher)
    assert teacher.apply_chat_template_kwargs == {"a": 1}
    params = inspect.signature(RepromptTeacher.__init__).parameters
    options = {k: v for k, v in cfg.teacher.items() if k != "_target_"}
    assert set(options) == set(params) - {"self", "tokenizer", "max_prefix_len", "apply_chat_template_kwargs",
                                          "success_reward_threshold"}
    # empty in the yaml, None in the signature: the reprompt renders with the dataset's kwargs
    assert options.pop("chat_template_kwargs") == {} and params["chat_template_kwargs"].default is None
    assert teacher.template_kwargs == {"a": 1}
    for name, value in options.items():
        assert getattr(teacher, name) == value
        assert params[name].default == value, f"{name}: reprompt.yaml and the constructor default differ"


def _compose_teacher_block(overrides):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="sdpo", overrides=overrides)
    return OmegaConf.to_container(cfg.actor_rollout_ref.actor.self_distillation, resolve=True)


def test_sdpo_config_composes_the_reprompt_teacher_with_only_its_own_keys():
    """``--config-name sdpo`` merges its own body after the ``sdpo_teacher`` group, so a
    teacher-specific key there would land on whichever teacher is selected."""
    reprompt = yaml.safe_load((CONFIG_DIR / "sdpo_teacher" / "reprompt.yaml").read_text())

    sd = _compose_teacher_block([])
    assert sd["teacher"] == reprompt and sd["teacher"]["max_reprompt_len"] == 10240
    teacher = make_teacher(OmegaConf.create(sd), ToyTokenizer(), max_prefix_len=4096)
    assert isinstance(teacher, RepromptTeacher) and teacher.max_reprompt_len == 10240


class TQStub:
    def __init__(self, data):
        self.data = data
        self.select_fields = None
        self.put = None

    def kv_batch_get(self, keys, partition_id, select_fields):
        self.select_fields = list(select_fields)
        return {k: self.data[k] for k in select_fields}

    def kv_batch_put(self, keys, partition_id, fields):
        assert isinstance(fields, TensorDict)
        self.put = (list(keys), fields)


TIMINGS = dict(
    loop_wall=10.0, generate_sequences=4.0, tool_calls=2.0, env_setup=1.0, reward_eval=0.5, reflect=0.25,
    num_preempted=1, eval_completed=1, capped_turns=0,
)


def test_trainer_reprompt_batch_fields_and_metrics(monkeypatch):
    """One trainer build over a batch with a failed row whose sibling solved the task, that
    solved sibling, a row with no extra_fields and blank feedback, and a failed row with
    feedback and no solution."""
    # key, uid, reward, feedback, extra_fields
    rows = [
        ("u1_0_0", "u1", 0.0, "fb0", dict(turn_spans=SPANS, segment_index=0, num_segments=1,
                                          traj_exit_reason="finished", timings=TIMINGS)),
        ("u1_1_0", "u1", 1.0, None, dict(turn_spans=SPANS, segment_index=0, num_segments=1,
                                         traj_exit_reason="submitted")),
        ("u2_0_0", "u2", 0.0, "   ", None),
        ("u2_1_0", "u2", 0.0, "fb3", dict(turn_spans=SPANS, segment_index=0, num_segments=1,
                                          traj_exit_reason="finished")),
    ]
    keys = [r[0] for r in rows]
    n = len(rows)
    inputs = _inputs([{} for _ in rows], [r[1] for r in rows], [r[2] for r in rows], [None] * n)
    rm_scores = []
    for r in rows:
        score = torch.zeros(RESPONSE.shape[0], dtype=torch.float32)
        score[-1] = r[2]
        rm_scores.append(score)
    extra_fields = [None if ef is None else dict(ef, reward_extra_info={"feedback": fb}) for _, _, _, fb, ef in rows]
    data = {
        "responses": torch.nested.nested_tensor(inputs.responses, layout=torch.jagged),
        "response_mask": torch.nested.nested_tensor(inputs.response_mask, layout=torch.jagged),
        "rm_scores": torch.nested.nested_tensor(rm_scores, layout=torch.jagged),
        "uid": inputs.uids,
        "raw_prompt": inputs.raw_prompts,
        "extra_fields": extra_fields,
    }
    stub = TQStub(data)
    monkeypatch.setattr(main_ppo_sync, "tq", stub)

    sd = OmegaConf.create(asdict(SelfDistillationConfig(
        success_reward_threshold=0.5,
        include_environment_feedback=True,
        teacher=dict(
            _target_=REPROMPT_TARGET,
            dont_reprompt_on_self_success=True,
            environment_feedback_only_without_solution=True,
        ),
    )))
    tok = ToyTokenizer()
    trainer = object.__new__(main_ppo_sync.PPOTrainer)
    trainer.config = OmegaConf.create(
        {"actor_rollout_ref": {"actor": {"policy_loss": {"loss_mode": "sdpo"}, "self_distillation": sd}}}
    )
    trainer.tokenizer = tok
    trainer.sdpo_teacher = make_teacher(sd, tok, max_prefix_len=4096)
    metrics = {}
    trainer._maybe_build_self_distillation_batch(SimpleNamespace(keys=keys, partition_id="train"), metrics)

    assert tok.decode_calls == 1, "only the solved sibling is decoded"
    assert stub.select_fields == ["responses", "rm_scores", "raw_prompt", "uid", "extra_fields", "response_mask"]
    put_keys, fields = stub.put
    assert put_keys == keys
    assert set(fields.keys()) == {"teacher_input_ids", "self_distillation_mask", "loss_mask", "trace_weight", "traj_id"}
    # row 0 learns from its sibling's solution (feedback dropped: only_without_solution),
    # row 3 from its feedback; the solved sibling and the blank-feedback row are unsupervised
    teacher = trainer.sdpo_teacher
    solution = teacher.solution_template.format(successful_previous_attempt=TURN0 + OBS + TURN1)
    reprompt0 = teacher.reprompt_template.format(prompt="task 0", solution=solution, feedback="")
    feedback3 = teacher.feedback_template.format(feedback_raw="fb3")
    reprompt3 = teacher.reprompt_template.format(prompt="task 3", solution="", feedback=feedback3)
    assert tok.last_batch == [
        [{"role": "system", "content": "sys"}, {"role": "user", "content": reprompt0}],
        inputs.raw_prompts[1],
        inputs.raw_prompts[2],
        [{"role": "system", "content": "sys"}, {"role": "user", "content": reprompt3}],
    ]
    assert fields["self_distillation_mask"].tolist() == [1.0, 0.0, 0.0, 1.0]
    tokens = RESPONSE.shape[0] - 1
    assert [int(m.sum()) for m in fields["loss_mask"].unbind()] == [tokens, 0, 0, tokens]
    assert fields["traj_id"].squeeze(-1).tolist() == [0, 1, 2, 3]
    assert fields["trace_weight"].squeeze(-1).tolist() == pytest.approx([1.0, 0.0, 0.0, 1.0])

    generated = 3.0 * (len(TURN0) + len(TURN1))
    expected = {
        "self_distillation/rows_per_step": 4.0,
        "self_distillation/traces_per_step": 4.0,
        "self_distillation/segments_per_trace_max": 1.0,
        "self_distillation/supervised_segments_per_trace_max": 1.0,
        "self_distillation/unsupervised_row_fraction": 2 / 4,
        "self_distillation/unsupervised_row_tokens": 2.0 * tokens,
        "self_distillation/supervised_row_tokens": 2.0 * tokens,
        "self_distillation/reprompt_sample_fraction": 2 / 4,
        "rollout/generated_tokens": generated,
        "rollout/generated_tokens_per_trace": generated / 4,
        # u1 has a success; it serves its sibling but not itself
        "self_distillation/success_group_fraction": 1 / 2,
        "self_distillation/success_sample_fraction": 1 / 4,
        "self_distillation/feedback_available_fraction": 2 / 4,
        "self_distillation/feedback_used_fraction": 1 / 4,
        "rollout/condensed_trace_fraction": 0.0,
        "rollout/segments_per_trace": 1.0,
        "rollout/solve_rate_1seg": 1 / 4,
        "rollout/trace_fraction_1seg": 1.0,
        "rollout/exit_finished_fraction": 2 / 4,
        "rollout/solve_rate_exit_finished": 0.0,
        "rollout/exit_submitted_fraction": 1 / 4,
        "rollout/solve_rate_exit_submitted": 1.0,
        "rollout/turns_in_segment_0": float(len(SPANS)),
    }
    # the one trajectory with timings sets every mean, max and quantile
    total = TIMINGS["loop_wall"] + TIMINGS["env_setup"] + TIMINGS["reward_eval"] + TIMINGS["reflect"]
    unattributed = TIMINGS["loop_wall"] - TIMINGS["generate_sequences"] - TIMINGS["tool_calls"]
    for key in ("generate_sequences", "tool_calls", "condense", "parse_action", "tokenize_observations",
                "loop_wall", "env_setup", "reward_eval", "reflect"):
        expected[f"traj_time/{key}_mean"] = float(TIMINGS.get(key, 0.0))
        expected[f"traj_time/slowest_{key}"] = float(TIMINGS.get(key, 0.0))
    expected.update({
        "rollout/preempted_reported_fraction": 1.0,
        "rollout/preempted_mean": 1.0,
        "rollout/preempted_max": 1.0,
        "rollout/preempted_trace_fraction": 1.0,
        "traj_time/unattributed_mean": unattributed,
        "traj_time/total_mean": total,
        "traj_time/total_max": total,
        "traj_time/total_p50": total,
        "traj_time/total_p90": total,
        "traj_time/unattributed_share": unattributed / total,
        "reward_health/eval_completed_fraction": 1.0,
        "reward_health/capped_turns_mean": 0.0,
        "reward_health/capped_rollouts_fraction": 0.0,
    })
    assert metrics == pytest.approx(expected)


class TQRoundTrip(TQStub):
    """The put fields are readable again, the way the update step reads the teacher's."""

    def kv_batch_put(self, keys, partition_id, fields):
        super().kv_batch_put(keys, partition_id, fields)
        self.data.update({k: fields[k] for k in fields.keys()})


def _traj_mean_trainer(monkeypatch, ppo_mini_batch_size=4, ppo_epochs=1):
    """A two-segment failed trajectory, a solved one and a single-segment failed one, built the
    way the trainer does under loss_agg_mode=traj-mean-token-mean."""
    short = RESPONSE[: len(TURN0)].clone()
    rows = [
        ("u1_0_0", "u1", 0.0, "fb", RESPONSE.clone(), dict(turn_spans=SPANS, segment_index=0, num_segments=2)),
        ("u1_0_1", "u1", 0.0, "fb", short,
         dict(turn_spans=[[0, 0, len(TURN0)]], segment_index=1, num_segments=2, segment_prompt=SEGMENT_PROMPT)),
        ("u2_0_0", "u2", 1.0, None, RESPONSE.clone(), dict(turn_spans=SPANS, segment_index=0, num_segments=1)),
        ("u3_0_0", "u3", 0.0, "fb3", RESPONSE.clone(), dict(turn_spans=SPANS, segment_index=0, num_segments=1)),
    ]
    keys = [r[0] for r in rows]
    inputs = _inputs([r[5] for r in rows], [r[1] for r in rows], [r[2] for r in rows], [None] * len(rows),
                     responses=[r[4] for r in rows])
    rm_scores = []
    for r in rows:
        score = torch.zeros(r[4].shape[0], dtype=torch.float32)
        score[-1] = r[2]
        rm_scores.append(score)
    data = {
        "responses": torch.nested.nested_tensor(inputs.responses, layout=torch.jagged),
        "response_mask": torch.nested.nested_tensor(inputs.response_mask, layout=torch.jagged),
        "rm_scores": torch.nested.nested_tensor(rm_scores, layout=torch.jagged),
        "uid": inputs.uids,
        "raw_prompt": inputs.raw_prompts,
        "extra_fields": [dict(ef, reward_extra_info={"feedback": fb}) for _, _, _, fb, _, ef in rows],
    }
    stub = TQRoundTrip(data)
    monkeypatch.setattr(main_ppo_sync, "tq", stub)

    sd = OmegaConf.create(asdict(SelfDistillationConfig(
        success_reward_threshold=0.5,
        include_environment_feedback=True,
        teacher=dict(_target_=REPROMPT_TARGET, dont_reprompt_on_self_success=True),
    )))
    tok = ToyTokenizer()
    trainer = object.__new__(main_ppo_sync.PPOTrainer)
    trainer.config = OmegaConf.create({
        "actor_rollout_ref": {
            "actor": {
                "policy_loss": {"loss_mode": "sdpo"},
                "self_distillation": sd,
                "loss_agg_mode": "traj-mean-token-mean",
                "ppo_mini_batch_size": ppo_mini_batch_size,
                "ppo_epochs": ppo_epochs,
                "calculate_entropy": False,
                "entropy_coeff": 0.0,
                "data_loader_seed": 1,
                "shuffle": True,
            },
            "rollout": {"n": 1, "temperature": 1.0},
        }
    })
    trainer.tokenizer = tok
    trainer.sdpo_teacher = make_teacher(sd, tok, max_prefix_len=4096)
    sent = []
    trainer.actor_rollout_wg = SimpleNamespace(update_actor=lambda b: sent.append(b) or {"metrics": {"mfu": 0.0}})
    batch = SimpleNamespace(keys=keys, partition_id="train", extra_info={})
    trainer._maybe_build_self_distillation_batch(batch, {})
    return trainer, stub, batch, sent


def test_trainer_traj_mean_shares_and_trajectory_count(monkeypatch):
    trainer, stub, batch, sent = _traj_mean_trainer(monkeypatch)
    _, fields = stub.put
    long_tokens, short_tokens = RESPONSE.shape[0] - 1, len(TURN0) - 1
    # raw shares: the failed trajectory's two segments split one unit by supervised tokens,
    # the solved trajectory has none, the single-segment failed one keeps the whole unit
    assert fields["trace_weight"].squeeze(-1).tolist() == pytest.approx(
        [long_tokens / (long_tokens + short_tokens), short_tokens / (long_tokens + short_tokens), 0.0, 1.0]
    )
    assert fields["traj_id"].squeeze(-1).tolist() == [0, 0, 1, 2]

    trainer._update_actor(batch, {})
    (update_batch,) = sent
    assert update_batch.extra_info["global_batch_size"] == 2, "two trajectories carry supervision"
    assert update_batch.extra_info["mini_batch_size"] == 4


def test_trainer_traj_mean_mini_batch_is_the_update_batch(monkeypatch):
    """The configured mini-batch size is not what the worker splits by: the whole update batch
    is one mini-batch, however many rows condensation left in it."""
    trainer, _, batch, sent = _traj_mean_trainer(monkeypatch, ppo_mini_batch_size=2)
    trainer._update_actor(batch, {})
    (update_batch,) = sent
    assert update_batch.extra_info["mini_batch_size"] == 4
    assert update_batch.extra_info["global_batch_size"] == 2


def test_trainer_traj_mean_skips_the_update_without_supervision(monkeypatch):
    trainer, stub, batch, sent = _traj_mean_trainer(monkeypatch)
    stub.data["trace_weight"] = torch.zeros_like(stub.data["trace_weight"])
    metrics = {}
    assert trainer._update_actor(batch, metrics) is batch
    assert sent == [], "no forward, no optimizer step"
    assert metrics == {"actor/skipped_update": 1.0}


def test_trainer_seq_mean_keeps_the_rescaled_weights_and_row_denominator(monkeypatch):
    trainer, stub, batch, sent = _traj_mean_trainer(monkeypatch)
    trainer.config.actor_rollout_ref.actor.loss_agg_mode = "seq-mean-token-mean"
    trainer._maybe_build_self_distillation_batch(batch, {})
    _, fields = stub.put
    # three supervised rows over two supervised trajectories: the raw shares scaled by 3/2
    assert sum(fields["trace_weight"].squeeze(-1).tolist()) == pytest.approx(3.0)
    trainer._update_actor(batch, {})
    (update_batch,) = sent
    assert update_batch.extra_info["global_batch_size"] == 4
