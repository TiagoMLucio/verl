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
"""SDPO teacher construction: the teacher contract, the batch a teacher reads and the fields
it hands back, and the paper's reprompt teacher.

``teacher`` is the contract (:class:`SDPOTeacher`, :func:`make_teacher`), ``batch`` what every
teacher reads and returns, ``teacher_meta`` the wire codec of spliced teacher rows, ``reprompt``
the paper path (:class:`RepromptTeacher`). ``self_distillation.teacher`` names the class by its
``_target_``; project teachers (the turn-hint teacher) live outside verl and subclass
:class:`SDPOTeacher`.
"""

from verl.trainer.ppo.sdpo.batch import TeacherBatch, TeacherInputs
from verl.trainer.ppo.sdpo.reprompt_teacher import RepromptTeacher
from verl.trainer.ppo.sdpo.teacher import SDPOTeacher, make_teacher

__all__ = ["RepromptTeacher", "SDPOTeacher", "TeacherBatch", "TeacherInputs", "make_teacher"]
