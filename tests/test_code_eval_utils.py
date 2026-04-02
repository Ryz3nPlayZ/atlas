from __future__ import annotations

from code_eval_utils import repair_body_completion


def test_repair_body_completion_keeps_longest_parsable_prefix() -> None:
    prompt = "def foo(x):\n"
    completion = "    value = x + 1\n    if value > 1:\n        return value\n    for\n"

    repaired = repair_body_completion(prompt, completion)

    assert repaired == "    value = x + 1\n    if value > 1:\n        return value\n"


def test_repair_body_completion_adds_pass_for_bare_block() -> None:
    prompt = "def foo(x):\n"
    completion = "    if x > 0:\n"

    repaired = repair_body_completion(prompt, completion)

    assert repaired == "    if x > 0:\n        pass\n"


def test_repair_body_completion_handles_prompt_with_trailing_indent() -> None:
    prompt = "def foo(x):\n    "
    completion = "    if x > 0:\n"

    repaired = repair_body_completion(prompt, completion)

    assert repaired == "    if x > 0:\n        pass\n"
