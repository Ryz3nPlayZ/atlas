from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_supervised_code_task_mix_v6.py"
SPEC = spec_from_file_location("prepare_supervised_code_task_mix_v6", SCRIPT_PATH)
MODULE = module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

build_codesearchnet_task_examples = MODULE.build_codesearchnet_task_examples
extract_imports_signature_body = MODULE.extract_imports_signature_body


def test_extract_imports_signature_body_strips_leading_docstring() -> None:
    code = """
import math

def area(radius):
    \"\"\"Compute the circle area.\"\"\"
    value = math.pi * radius * radius
    return value
""".strip()

    imports, signature, body = extract_imports_signature_body(code)

    assert imports == ["import math"]
    assert signature == "def area(radius):"
    assert '"""' not in body
    assert "return value" in body


def test_build_codesearchnet_task_examples_filters_methods_and_emits_prompt_completion() -> None:
    method_code = """
def save(self, path):
    \"\"\"Save the object.\"\"\"
    return path
""".strip()
    assert build_codesearchnet_task_examples(method_code, "Save the object.", 200, 20, 400) == []

    function_code = """
def add(a, b):
    \"\"\"Return the sum of two integers.\"\"\"
    return a + b
""".strip()
    examples = build_codesearchnet_task_examples(function_code, "Return the sum of two integers.", 200, 20, 400)

    assert len(examples) == 2
    assert all(example["prompt"].endswith("\n    ") for example in examples)
    assert all(example["completion"].startswith("    return a + b") for example in examples)
