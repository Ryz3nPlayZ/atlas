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
        description="Prepare a task-aligned code finetuning mix with MBPP-style body completion."
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "task_aligned_code_ft")
    parser.add_argument("--codesearchnet-limit", type=int, default=10000)
    parser.add_argument("--codesearchnet-validation-limit", type=int, default=1000)
    parser.add_argument("--mbpp-repeat", type=int, default=200)
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
        if not in_signature and not signature_done and (stripped.startswith("import ") or stripped.startswith("from ")):
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


def build_mbpp_prompt_example(prompt: str, code: str) -> list[str]:
    imports, signature, body = extract_imports_signature_body(code)
    if not signature or not body:
        return []
    prompt_style = [f"# Task: {prompt.strip()}"]
    if imports:
        prompt_style.extend(imports)
    prompt_style.append(signature)
    prompt_style.append(indent_block(body))

    humaneval_style = []
    if imports:
        humaneval_style.extend(imports)
    humaneval_style.append(signature)
    humaneval_style.append('    """')
    humaneval_style.append(f"    {prompt.strip()}")
    humaneval_style.append('    """')
    humaneval_style.append(indent_block(body))

    return ["\n".join(prompt_style) + "\n", "\n".join(humaneval_style) + "\n"]


def build_codesearchnet_prompt_example(code: str) -> str | None:
    imports, signature, body = extract_imports_signature_body(code)
    if not signature or not body:
        return None
    lines = ["# Complete the Python function body."]
    if imports:
        lines.extend(imports)
    lines.append(signature)
    lines.append(indent_block(body))
    return "\n".join(lines) + "\n"


def write_jsonl(records: list[str], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for text in records:
            handle.write(json.dumps({"text": text}) + "\n")
    return len(records)


def write_tokenized(jsonl_path: Path, token_path: Path, tokenizer: ByteTokenizer) -> int:
    count = 0
    with jsonl_path.open("r", encoding="utf-8") as source, token_path.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            sink.write(json.dumps({"tokens": tokenizer.encode(text)}) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteTokenizer()

    train_records: list[str] = []
    val_records: list[str] = []

    mbpp_train = load_dataset_split("mbpp", "train", "sanitized")
    mbpp_val = load_dataset_split("mbpp", "validation", "sanitized")
    mbpp_prompt = load_dataset_split("mbpp", "prompt", "sanitized")
    for split_rows in (mbpp_train, mbpp_val, mbpp_prompt):
        for row in split_rows:
            examples = build_mbpp_prompt_example(row["prompt"], row["code"])
            for _ in range(args.mbpp_repeat):
                train_records.extend(examples)

    codesearchnet_train = load_dataset_split(
        "code_search_net",
        f"train[:{args.codesearchnet_limit}]",
        "python",
    )
    for row in codesearchnet_train:
        example = build_codesearchnet_prompt_example(str(row.get("func_code_string", "")).strip())
        if example is not None:
            train_records.append(example)

    codesearchnet_val = load_dataset_split(
        "code_search_net",
        f"validation[:{args.codesearchnet_validation_limit}]",
        "python",
    )
    for row in codesearchnet_val:
        example = build_codesearchnet_prompt_example(str(row.get("func_code_string", "")).strip())
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
