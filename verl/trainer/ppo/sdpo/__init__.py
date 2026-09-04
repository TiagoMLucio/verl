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
"""SDPO teacher construction: the two teacher classes the trainer can run, the batch they read
and the fields they hand back.

``hints``, ``splice`` and ``teacher_meta`` are the turn-hint path (:class:`TurnHintTeacher`);
``reprompt`` is the paper path (:class:`RepromptTeacher`). ``cfg.teacher`` picks the class.
"""

from verl.trainer.ppo.sdpo.batch import TeacherBatch, TeacherInputs
from verl.trainer.ppo.sdpo.reprompt_teacher import RepromptTeacher
from verl.trainer.ppo.sdpo.turn_hint_teacher import TurnHintTeacher

__all__ = ["RepromptTeacher", "TeacherBatch", "TeacherInputs", "TurnHintTeacher", "make_teacher"]


def make_teacher(cfg, tokenizer, apply_chat_template_kwargs=None, max_prompt_length=None):
    """The teacher ``cfg.teacher`` names; ``max_prompt_length`` caps the turn-hint prefix."""
    if cfg.teacher == "turn_hints":
        return TurnHintTeacher(tokenizer, cfg, max_prefix_len=max_prompt_length)
    return RepromptTeacher(tokenizer, cfg, apply_chat_template_kwargs)
