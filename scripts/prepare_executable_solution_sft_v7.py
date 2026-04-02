from __future__ import annotations

import argparse
import ast
import json
import textwrap
import warnings
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ace_atlas.tokenizer.byte_level import ByteTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a benchmark-style executable-solution SFT mix using MBPP plus "
            "held-out HumanEval training tasks."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "executable_solution_sft_v7")
    parser.add_argument("--mbpp-repeat", type=int, default=300)
    parser.add_argument("--humaneval-repeat", type=int, default=250)
    parser.add_argument("--max-tests", type=int, default=3)
    parser.add_argument("--humaneval-eval-tasks", type=int, default=10)
    parser.add_argument("--humaneval-val-tasks", type=int, default=10)
    return parser.parse_args()


def load_dataset_split(name: str, split: str, config: str | None = None):
    from datasets import load_dataset

    if config is None:
        return load_dataset(name, split=split)
    return load_dataset(name, config, split=split)


def is_docstring_expr(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(getattr(node, "value", None), ast.Constant)
        and isinstance(node.value.value, str)
    )


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
    if not signature:
        return imports, "", body

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            module = ast.parse(code)
        function_node = next(
            node for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
    except (SyntaxError, StopIteration):
        return imports, signature, body

    if not function_node.body:
        return imports, signature, body

    start_stmt = function_node.body[0]
    if is_docstring_expr(start_stmt):
        if len(function_node.body) == 1:
            return imports, signature, ""
        start_stmt = function_node.body[1]

    start_line = getattr(start_stmt, "lineno", None)
    end_line = getattr(function_node, "end_lineno", None)
    if start_line is None or end_line is None:
        return imports, signature, body

    stripped_body = textwrap.dedent("\n".join(lines[start_line - 1 : end_line])).strip("\n")
    return imports, signature, stripped_body


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

    examples: list[dict[str, str]] = []
    prompt_lines = [
        "# Write only the Python function body for the signature below.",
        f"# Task: {prompt.strip()}",
    ]
    if imports:
        prompt_lines.extend(imports)
    prompt_lines.append(signature)
    examples.append({"prompt": "\n".join(prompt_lines) + "\n    ", "completion": completion})

    prompt_with_tests = [
        "# Write only the Python function body for the signature below.",
        f"# Task: {prompt.strip()}",
        *render_tests(test_list, max_tests=max_tests),
    ]
    if imports:
        prompt_with_tests.extend(imports)
    prompt_with_tests.append(signature)
    examples.append({"prompt": "\n".join(prompt_with_tests) + "\n    ", "completion": completion})

    docstring_prompt = []
    if imports:
        docstring_prompt.extend(imports)
    docstring_prompt.append(signature)
    docstring_prompt.append('    """')
    docstring_prompt.append(f"    {prompt.strip()}")
    docstring_prompt.append('    """')
    examples.append({"prompt": "\n".join(docstring_prompt) + "\n    ", "completion": completion})

    return examples


def build_humaneval_example(prompt: str, canonical_solution: str) -> dict[str, str]:
    completion = canonical_solution
    if not completion.endswith("\n"):
        completion += "\n"
    if not completion.startswith("    "):
        completion = indent_block(completion.lstrip("\n")) + "\n"
    prompt_text = "# Complete the Python function below.\n" + prompt.rstrip() + "    "
    return {"prompt": prompt_text, "completion": completion}


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

    for split_name in ("train", "prompt", "validation"):
        split_rows = load_dataset_split("mbpp", split_name, "sanitized")
        for row in split_rows:
            examples = build_mbpp_examples(
                row["prompt"],
                row["code"],
                row.get("test_list", []),
                max_tests=args.max_tests,
            )
            for _ in range(args.mbpp_repeat):
                train_records.extend(examples)

    humaneval = list(load_dataset_split("openai_humaneval", "test"))
    eval_cutoff = args.humaneval_eval_tasks
    val_cutoff = eval_cutoff + args.humaneval_val_tasks

    humaneval_val = humaneval[eval_cutoff:val_cutoff]
    humaneval_train = humaneval[val_cutoff:]

    for row in humaneval_train:
        example = build_humaneval_example(row["prompt"], row["canonical_solution"])
        for _ in range(args.humaneval_repeat):
            train_records.append(example)

    for row in humaneval_val:
        val_records.append(build_humaneval_example(row["prompt"], row["canonical_solution"]))

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
