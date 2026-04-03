from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_mixed_code_text_base_v1.py"
SPEC = spec_from_file_location("prepare_mixed_code_text_base_v1", SCRIPT_PATH)
MODULE = module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

read_jsonl = MODULE.read_jsonl


def test_read_jsonl_reads_non_empty_lines(tmp_path: Path) -> None:
    path = tmp_path / "tokens.jsonl"
    path.write_text('{"tokens":[1,2,3]}\n\n{"tokens":[4,5,6]}\n', encoding="utf-8")

    rows = read_jsonl(path)

    assert rows == [{"tokens": [1, 2, 3]}, {"tokens": [4, 5, 6]}]
