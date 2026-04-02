from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ace_atlas.tokenizer.byte_level import ByteTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a strongly supervised code-task finetuning mix with MBPP-style answer spans."
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "supervised_code_task_ft")
    parser.add_argument("--codesearchnet-support-limit", type=int, default=2000)
    parser.add_argument("--codesearchnet-validation-limit", type=int, default=250)
    parser.add_argument("--mbpp-repeat", type=int, default=400)
    parser.add_argument("--max-tests", type=int, default=2)
    return parser.parse_args()


def load_dataset_split(name: str, split: str, config: str | None = None):
    from datasets import load_dataset

    if config is None:
        return load_dataset(name, split=split)
    return load_dataset(name, config, split=split)


def extract_imports_signature_body(code: str) -> tuple[list[str], str, str]:
    lines = code.splitlines()
    imports: list[str] = []
    signature_lines: list[str] = []
    body_lines: list[str] = []
    in_signature = False
    signature_done = False

    for line in lines:
        stripped = line.strip()
        if not in_signature and not signature_done and (
            stripped.startswith("import ") or stripped.startswith("from ")
        ):
            imports.append(line)
            continue
        if not signature_done and stripped.startswith("def "):
            in_signature = True
            signature_lines.append(line)
            if stripped.endswith(":"):
                in_signature = False
                signature_done = True
            continue
        if in_signature:
            signature_lines.append(line)
            if stripped.endswith(":"):
                in_signature = False
                signature_done = True
            continue
        if signature_done:
            body_lines.append(line)

    signature = "\n".join(signature_lines).rstrip()
    body = textwrap.dedent("\n".join(body_lines)).strip("\n")
    return imports, signature, body


def indent_block(text: str, indent: str = "    ") -> str:
    text = text.strip("\n")
    if not text.strip():
        return ""
    return "\n".join(f"{indent}{line}" if line.strip() else "" for line in text.splitlines())


def render_tests(test_list: list[str], max_tests: int) -> list[str]:
    rendered: list[str] = ["# Example tests:"]
    for test in test_list[:max_tests]:
        for line in test.splitlines():
            rendered.append(f"# {line.rstrip()}")
    return rendered


def build_mbpp_examples(prompt: str, code: str, test_list: list[str], max_tests: int) -> list[dict[str, str]]:
    imports, signature, body = extract_imports_signature_body(code)
    if not signature or not body:
        return []
    completion = indent_block(body) + "\n"

    task_lines = [
        "# Write only the Python function body for the signature below.",
        f"# Task: {prompt.strip()}",
    ]
    if imports:
        task_lines.extend(imports)
    task_lines.append(signature)

    task_with_tests_lines = [
        "# Write only the Python function body for the signature below.",
        f"# Task: {prompt.strip()}",
        *render_tests(test_list, max_tests=max_tests),
    ]
    if imports:
        task_with_tests_lines.extend(imports)
    task_with_tests_lines.append(signature)

    docstring_lines = []
    if imports:
        docstring_lines.extend(imports)
    docstring_lines.append(signature)
    docstring_lines.append('    """')
    docstring_lines.append(f"    {prompt.strip()}")
    docstring_lines.append('    """')

    docstring_with_tests_lines = []
    if imports:
        docstring_with_tests_lines.extend(imports)
    docstring_with_tests_lines.append(signature)
    docstring_with_tests_lines.append('    """')
    docstring_with_tests_lines.append(f"    {prompt.strip()}")
    docstring_with_tests_lines.append("")
    docstring_with_tests_lines.append("    Example tests:")
    for test in test_list[:max_tests]:
        docstring_with_tests_lines.append(f"    {test.strip()}")
    docstring_with_tests_lines.append('    """')

    return [
        {"prompt": "\n".join(task_lines) + "\n    ", "completion": completion},
        {"prompt": "\n".join(task_with_tests_lines) + "\n    ", "completion": completion},
        {"prompt": "\n".join(docstring_lines) + "\n    ", "completion": completion},
        {"prompt": "\n".join(docstring_with_tests_lines) + "\n    ", "completion": completion},
    ]


def build_codesearchnet_support_example(code: str) -> dict[str, str] | None:
    imports, signature, body = extract_imports_signature_body(code)
    if not signature or not body:
        return None
    lines = ["# Write only the Python function body for the signature below."]
    if imports:
        lines.extend(imports)
    lines.append(signature)
    return {"prompt": "\n".join(lines) + "\n    ", "completion": indent_block(body) + "\n"}


def write_jsonl(records: list[dict[str, str]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return len(records)


def write_tokenized(jsonl_path: Path, token_path: Path, tokenizer: ByteTokenizer) -> int:
    count = 0
    with jsonl_path.open("r", encoding="utf-8") as source, token_path.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = record["prompt"]
            completion = record["completion"]
            prompt_tokens = list(prompt.encode("utf-8"))
            completion_tokens = tokenizer.encode(completion)
            tokens = prompt_tokens + completion_tokens
            loss_mask = [0] * len(prompt_tokens) + [1] * len(completion_tokens)
            sink.write(json.dumps({"tokens": tokens, "loss_mask": loss_mask}) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteTokenizer()

    train_records: list[dict[str, str]] = []
    val_records: list[dict[str, str]] = []

    mbpp_train = load_dataset_split("mbpp", "train", "sanitized")
    mbpp_prompt = load_dataset_split("mbpp", "prompt", "sanitized")
    mbpp_val = load_dataset_split("mbpp", "validation", "sanitized")

    for split_rows in (mbpp_train, mbpp_prompt):
        for row in split_rows:
            examples = build_mbpp_examples(
                row["prompt"],
                row["code"],
                row.get("test_list", []),
                max_tests=args.max_tests,
            )
            for _ in range(args.mbpp_repeat):
                train_records.extend(examples)

    for row in mbpp_val:
        val_records.extend(
            build_mbpp_examples(
                row["prompt"],
                row["code"],
                row.get("test_list", []),
                max_tests=args.max_tests,
            )
        )

    codesearchnet_train = load_dataset_split(
        "code_search_net",
        f"train[:{args.codesearchnet_support_limit}]",
        "python",
    )
    for row in codesearchnet_train:
        example = build_codesearchnet_support_example(str(row.get("func_code_string", "")).strip())
        if example is not None:
            train_records.append(example)

    codesearchnet_val = load_dataset_split(
        "code_search_net",
        f"validation[:{args.codesearchnet_validation_limit}]",
        "python",
    )
    for row in codesearchnet_val:
        example = build_codesearchnet_support_example(str(row.get("func_code_string", "")).strip())
        if example is not None:
            val_records.append(example)

    train_jsonl = args.output_dir / "train.jsonl"
    val_jsonl = args.output_dir / "validation.jsonl"
    train_tokens = args.output_dir / "train_tokens.jsonl"
    val_tokens = args.output_dir / "validation_tokens.jsonl"

    train_count = write_jsonl(train_records, train_jsonl)
    val_count = write_jsonl(val_records, val_jsonl)
    train_token_count = write_tokenized(train_jsonl, train_tokens, tokenizer)
    val_token_count = write_tokenized(val_jsonl, val_tokens, tokenizer)

    print(f"Wrote train JSONL: {train_count} rows -> {train_jsonl}")
    print(f"Wrote train tokenized: {train_token_count} rows -> {train_tokens}")
    print(f"Wrote validation JSONL: {val_count} rows -> {val_jsonl}")
    print(f"Wrote validation tokenized: {val_token_count} rows -> {val_tokens}")


if __name__ == "__main__":
    main()
