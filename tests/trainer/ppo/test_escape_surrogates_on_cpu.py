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
"""Undecodable bytes in a tool or test output reach an agent loop's fields as lone surrogates. msgspec rejects
them, and the pickle TransferQueue 0.1.7 then falls back to kills the storage unit that receives it, so the
fields are escaped before the put."""

import pytest
import torch

from verl.trainer.main_ppo_sync import _escape_surrogates

BAD = b"bad_\xff_name.txt".decode("utf-8", "surrogateescape")
ESCAPED = "bad_\\udcff_name.txt"


def _field():
    return {
        "extra_fields": {
            "reward_extra_info": {"feedback": "FAILED test_file_error_surrogates: " + BAD},
            "turn_hints": [[23, BAD]],
        },
        "raw_prompt": ({"role": "user", "content": BAD},),
        "responses": torch.arange(4),
    }


def test_lone_surrogates_are_escaped_wherever_they_sit():
    escaped = _escape_surrogates(_field())
    assert escaped["extra_fields"]["reward_extra_info"]["feedback"] == "FAILED test_file_error_surrogates: " + ESCAPED
    assert escaped["extra_fields"]["turn_hints"] == [[23, ESCAPED]]
    assert escaped["raw_prompt"] == ({"role": "user", "content": ESCAPED},)
    assert escaped["extra_fields"]["reward_extra_info"]["feedback"].encode("utf-8")


def test_clean_values_pass_through_as_the_same_objects():
    responses, ids, text = torch.arange(4), list(range(100_000)), "plain text"
    assert _escape_surrogates(responses) is responses
    assert _escape_surrogates(ids) is ids
    assert _escape_surrogates(text) is text


def test_msgspec_encodes_the_escaped_field_and_rejects_the_raw_one():
    msgpack = pytest.importorskip("msgspec.msgpack")
    field = {k: v for k, v in _field().items() if k != "responses"}
    with pytest.raises(UnicodeEncodeError):
        msgpack.encode(field)
    assert msgpack.decode(msgpack.encode(_escape_surrogates(field)))["raw_prompt"][0]["content"] == ESCAPED
