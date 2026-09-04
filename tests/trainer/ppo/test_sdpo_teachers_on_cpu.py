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
"""The two SDPO teachers on a toy tokenizer: the turn-hint teacher's fields agree with the
splice called directly and decode nothing; the reprompt teacher builds the paper's messages,
masks whole rows and decodes only the responses it uses as a solution; and the trainer's
teacher-build step writes the six fields and the batch metrics a turn_hints run logs."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.trainer import main_ppo_sync
from verl.trainer.ppo.sdpo import RepromptTeacher, TeacherInputs, TurnHintTeacher, make_teacher
from verl.trainer.ppo.sdpo.hints import assistant_header_ids, hint_token_ids, select_hinted_turns
from verl.trainer.ppo.sdpo.splice import build_spliced_teacher_row, turn_token_mask
from verl.trainer.ppo.sdpo.teacher_meta import DEGENERATE_META
from verl.workers.config.actor import SelfDistillationConfig


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
        keys=[f"{uid}_0_{i}" for i, uid in enumerate(uids)],
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
    )


def test_turn_hint_teacher_matches_the_splice_and_decodes_nothing():
    tok = ToyTokenizer()
    cfg = SelfDistillationConfig(teacher="turn_hints", max_hinted_turns=None)
    teacher = make_teacher(cfg, tok, max_prompt_length=4096)
    assert isinstance(teacher, TurnHintTeacher) and teacher.needs_prompts
    extra = [
        {"turn_spans": SPANS, "turn_feedback": [[0, "h0"], [1, "h1", "call"]]},
        {"turn_spans": SPANS, "turn_feedback": []},
        {"turn_spans": SPANS, "turn_feedback": [[1, "h2"]], "segment_index": 1, "segment_prompt": SEGMENT_PROMPT},
    ]
    inputs = _inputs(extra, ["a", "a", "b"], [0.0, 0.0, 0.0], [None] * 3)

    out = teacher.build(inputs)

    assert tok.decode_calls == 0
    assert set(out.fields) == {"teacher_input_ids", "teacher_seq_meta", "self_distillation_mask", "loss_mask"}
    assert out.hinted_per_row == [select_hinted_turns(ef, RESPONSE.shape[0]) for ef in extra]
    assert [len(h) for h in out.hinted_per_row] == [2, 0, 1]
    assert [h.is_call for h in out.hinted_per_row[0]] == [False, True]

    header = torch.tensor(assistant_header_ids(tok), dtype=torch.int64)
    assert torch.equal(header, ids(HEADER))
    for row in (0, 2):
        hinted = out.hinted_per_row[row]
        seq, meta, fallbacks, spans = build_spliced_teacher_row(
            PROMPT, RESPONSE, hinted, [hint_token_ids(tok, h, cfg) for h in hinted], 4096, header,
            close_ids=ids("<eos>\n"), call_open_ids=ids("<tool_call>"),
        )
        assert fallbacks == 0
        assert torch.equal(out.fields["teacher_input_ids"][row], seq)
        assert out.fields["teacher_seq_meta"][row].tolist() == meta
        mask = turn_token_mask(RESPONSE.shape[0], spans)
        assert torch.equal(out.fields["self_distillation_mask"][row], mask)
        assert torch.equal(out.fields["loss_mask"][row], inputs.response_mask[row] * mask.to(torch.int64))
    assert out.fields["loss_mask"][0][1] == 0, "observation tokens stay out of the loss"

    assert out.fields["teacher_seq_meta"][1].tolist() == DEGENERATE_META
    assert torch.equal(out.fields["teacher_input_ids"][1], torch.cat([PROMPT[-1:], RESPONSE[:1]]))
    assert out.fields["self_distillation_mask"][1].sum() == 0 and out.fields["loss_mask"][1].sum() == 0

    assert out.metrics == {
        "self_distillation/hinted_sample_fraction": 2 / 3,
        "self_distillation/hinted_turns_per_sample": 1.5,
        "self_distillation/hint_injection_fallbacks": 0,
        "self_distillation/call_loss_weight": 1.0,
    }


def test_reprompt_teacher_messages_masks_and_lazy_decode():
    tok = ToyTokenizer()
    cfg = SelfDistillationConfig(
        teacher="reprompt",
        include_environment_feedback=True,
        dont_reprompt_on_self_success=True,
        remove_thinking_from_demonstration=True,
        max_reprompt_len=512,
        reprompt_truncation="left",
    )
    teacher = make_teacher(cfg, tok, apply_chat_template_kwargs={})
    assert isinstance(teacher, RepromptTeacher) and not teacher.needs_prompts
    # uid a: row 0 failed with feedback, row 1 solved (its solution serves row 0, not itself);
    # uid b: row 2 failed without feedback, row 3 a condensation segment with feedback
    extra = [{}, {}, {}, {"segment_index": 1, "segment_prompt": SEGMENT_PROMPT}]
    responses = [RESPONSE.clone(), ids("<think>t</think>sol"), RESPONSE.clone(), RESPONSE.clone()]
    inputs = _inputs(extra, ["a", "a", "b", "b"], [0.0, 1.0, 0.0, 0.0], ["fb0", None, None, "fb3"], responses)

    out = teacher.build(inputs)

    assert tok.decode_calls == 1, "only the response used as a solution is decoded"
    assert (tok.padding_side, tok.truncation_side) == ("right", "right"), "tokenizer sides are restored"
    assert set(out.fields) == {"teacher_input_ids", "self_distillation_mask", "loss_mask"}
    assert out.hinted_per_row is None and out.metrics == {}

    solution = cfg.solution_template.format(successful_previous_attempt="sol")
    feedback0 = cfg.feedback_template.format(feedback_raw="fb0")
    feedback3 = cfg.feedback_template.format(feedback_raw="fb3")
    reprompt0 = cfg.reprompt_template.format(prompt="task 0", solution=solution, feedback=feedback0)
    reprompt3 = cfg.reprompt_template.format(prompt="", solution="", feedback=feedback3)
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
    cfg = SelfDistillationConfig(teacher="reprompt", max_reprompt_len=8, reprompt_truncation="left")
    inputs = _inputs([{}], ["a"], [1.0], [None])
    out = RepromptTeacher(tok, cfg).build(inputs)
    prompt = ids(ToyTokenizer.render(inputs.raw_prompts[0], add_generation_prompt=True))
    assert torch.equal(out.fields["teacher_input_ids"][0][:8], prompt[-8:]), "left-truncated to max_reprompt_len"
    assert tok.truncation_side == "right"


def test_turn_hint_teacher_counts_one_fallback_per_hint():
    tok = ToyTokenizer()
    cfg = SelfDistillationConfig(teacher="turn_hints")
    teacher = make_teacher(cfg, tok, max_prompt_length=4096)
    mid_turn = [[0, 1, len(TURN0)], SPANS[1]]
    extra = [
        # a call hint on a turn without <tool_call> whose span also starts mid-turn: one fallback
        {"turn_spans": mid_turn, "turn_feedback": [[0, "h0", "call"], [1, "h1"]]},
        # a turn hint whose span start is not preceded by the assistant header
        {"turn_spans": mid_turn, "turn_feedback": [[0, "h2"]]},
        {"turn_spans": SPANS, "turn_feedback": [[0, "h3"], [1, "h4", "call"]]},
    ]
    out = teacher.build(_inputs(extra, ["a", "b", "c"], [0.0] * 3, [None] * 3))
    assert out.metrics["self_distillation/hint_injection_fallbacks"] == 2
    assert [len(h) for h in out.hinted_per_row] == [2, 1, 2]
    assert torch.equal(out.fields["self_distillation_mask"][1][1 : len(TURN0)], torch.ones(len(TURN0) - 1))


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


def test_trainer_turn_hints_batch_fields_and_metrics(monkeypatch):
    """One trainer build over a batch with a call-hinted row, a condensed trajectory whose only
    hint sits on its second segment, an unhinted successful sibling, a row with no extra_fields
    and a row with a first-turn hint."""
    # key, uid, reward, feedback, extra_fields
    rows = [
        ("u1_0_0", "u1", 0.0, "fb0", dict(turn_spans=SPANS, turn_feedback=[[0, "h0"], [1, "h1", "call"]],
                                          segment_index=0, num_segments=1, traj_exit_reason="finished",
                                          timings=TIMINGS)),
        ("u1_1_0", "u1", 1.0, None, dict(turn_spans=SPANS, turn_feedback=[], segment_index=0, num_segments=2,
                                         traj_exit_reason="submitted")),
        ("u1_1_1", "u1", 1.0, None, dict(turn_spans=SPANS, turn_feedback=[[1, "h2"]], segment_index=1,
                                         num_segments=2, segment_prompt=SEGMENT_PROMPT)),
        ("u2_0_0", "u2", 0.0, "   ", None),
        ("u2_1_0", "u2", 0.0, "fb4", dict(turn_spans=SPANS, turn_feedback=[[0, "h3"]], segment_index=0,
                                          num_segments=1, traj_exit_reason="finished")),
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
        "prompts": torch.nested.nested_tensor(inputs.prompts, layout=torch.jagged),
        "rm_scores": torch.nested.nested_tensor(rm_scores, layout=torch.jagged),
        "uid": inputs.uids,
        "raw_prompt": inputs.raw_prompts,
        "extra_fields": extra_fields,
    }
    stub = TQStub(data)
    monkeypatch.setattr(main_ppo_sync, "tq", stub)

    sd = OmegaConf.create(asdict(SelfDistillationConfig(
        teacher="turn_hints", call_loss_weight=2.0, success_reward_threshold=0.5,
        include_environment_feedback=True, environment_feedback_only_without_solution=True,
    )))
    tok = ToyTokenizer()
    trainer = object.__new__(main_ppo_sync.PPOTrainer)
    trainer.config = OmegaConf.create(
        {"actor_rollout_ref": {"actor": {"policy_loss": {"loss_mode": "sdpo"}, "self_distillation": sd}}}
    )
    trainer.tokenizer = tok
    trainer.sdpo_teacher = make_teacher(sd, tok, max_prompt_length=4096)
    metrics = {}
    trainer._maybe_build_self_distillation_batch(SimpleNamespace(keys=keys, partition_id="train"), metrics)

    assert tok.decode_calls == 0
    assert stub.select_fields == [
        "responses", "rm_scores", "raw_prompt", "uid", "extra_fields", "response_mask", "prompts"
    ]
    put_keys, fields = stub.put
    assert put_keys == keys
    assert set(fields.keys()) == {
        "teacher_input_ids", "teacher_seq_meta", "self_distillation_mask", "loss_mask", "trace_weight", "traj_id"
    }
    # supervised tokens: turn 0 minus the observation token at index 1, plus the call span from <tool_call>
    call_span = len(TURN1) - TURN1.index("<tool_call>")
    supervised = [len(TURN0) - 1 + call_span, 0, len(TURN1), 0, len(TURN0) - 1]
    assert [int(m.sum()) for m in fields["loss_mask"].unbind()] == supervised
    assert fields["traj_id"].squeeze(-1).tolist() == [0, 1, 1, 2, 3]
    # raw shares (2.0 for the call row, 1, 1) renormalised to the three supervised rows
    assert fields["trace_weight"].squeeze(-1).tolist() == pytest.approx([1.5, 0.0, 0.75, 0.0, 0.75])
    assert fields["teacher_seq_meta"][1].tolist() == DEGENERATE_META
    assert fields["teacher_seq_meta"][3].tolist() == DEGENERATE_META

    tokens = RESPONSE.shape[0] - 1
    turns = len(SPANS)
    expected = {
        "self_distillation/rows_per_step": 5.0,
        "self_distillation/traces_per_step": 4.0,
        "self_distillation/segments_per_trace_max": 2.0,
        "self_distillation/supervised_segments_per_trace_max": 1.0,
        "self_distillation/unsupervised_row_fraction": 2 / 5,
        "self_distillation/unsupervised_row_tokens": 2.0 * tokens,
        "self_distillation/supervised_row_tokens": 3.0 * tokens,
        "self_distillation/reprompt_sample_fraction": 3 / 5,
        "rollout/generated_tokens": 4.0 * (len(TURN0) + len(TURN1)),
        "rollout/generated_tokens_per_trace": 1.0 * (len(TURN0) + len(TURN1)),
        # u1 has a success; first-segment rows are u1_0, u1_1, u2_0, u2_1
        "self_distillation/success_group_fraction": 1 / 2,
        "self_distillation/success_sample_fraction": 2 / 4,
        "self_distillation/feedback_available_fraction": 2 / 4,
        "self_distillation/feedback_used_fraction": 1 / 4,
        "self_distillation/hinted_sample_fraction": 3 / 5,
        "self_distillation/hinted_turns_per_sample": 4 / 3,
        "self_distillation/hint_injection_fallbacks": 0,
        "self_distillation/call_loss_weight": 2.0,
        "rollout/condensed_trace_fraction": 1 / 4,
        "rollout/segments_per_trace": 5 / 4,
        "rollout/solve_rate_1seg": 0.0,
        "rollout/trace_fraction_1seg": 3 / 4,
        "rollout/solve_rate_2seg": 1.0,
        "rollout/trace_fraction_2seg": 1 / 4,
        "rollout/exit_finished_fraction": 2 / 4,
        "rollout/solve_rate_exit_finished": 0.0,
        "rollout/exit_submitted_fraction": 1 / 4,
        "rollout/solve_rate_exit_submitted": 1.0,
        "rollout/turns_in_segment_0": float(turns),
        "rollout/turns_in_segment_1": float(turns),
        "self_distillation/hinted_trace_fraction": 3 / 4,
        "self_distillation/hinted_turns_per_trace": 4 / 3,
        "self_distillation/call_row_fraction": 1 / 3,
        "self_distillation/call_row_weight_share": 1.5 / 3.0,
        # hints at steps 0 and 1 of u1_0, 1 of u1_1, 0 of u2_1, every trajectory spanning steps 0..1
        "self_distillation/hint_position_mean": 0.5,
        "self_distillation/hint_position_median": 1.0,
        "self_distillation/hint_position_first_half": 0.5,
        "self_distillation/hint_in_last_two_turns": 1.0,
        "self_distillation/hint_gap_mean": 1.0,
        "self_distillation/hint_adjacent_fraction": 1.0,
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
    assert isinstance(metrics["self_distillation/hint_injection_fallbacks"], int)
