from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_executable_solution_sft_v7.py"
SPEC = spec_from_file_location("prepare_executable_solution_sft_v7", SCRIPT_PATH)
MODULE = module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

build_humaneval_example = MODULE.build_humaneval_example
extract_imports_signature_body = MODULE.extract_imports_signature_body


def test_build_humaneval_example_preserves_prompt_and_indents_completion() -> None:
    prompt = "def add(a, b):\n    \"\"\"Return sum.\"\"\"\n"
    solution = "    return a + b\n"

    record = build_humaneval_example(prompt, solution)

    assert record["prompt"].startswith("# Complete the Python function below.\n")
    assert record["prompt"].endswith("    ")
    assert record["completion"] == "    return a + b\n"


def test_extract_imports_signature_body_removes_docstring() -> None:
    code = """
from math import pi

def area(radius):
    \"\"\"Compute area.\"\"\"
    return pi * radius * radius
""".strip()

    imports, signature, body = extract_imports_signature_body(code)

    assert imports == ["from math import pi"]
    assert signature == "def area(radius):"
    assert '"""' not in body
    assert "return pi * radius * radius" in body
