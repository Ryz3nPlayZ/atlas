from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
import textwrap
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ace_atlas.config import ACEAtlasConfig
from ace_atlas.experiment import build_model, count_parameters, format_parameter_count
from ace_atlas.tokenizer.byte_level import ByteTokenizer


STOP_MARKERS = (
    "\ndef ",
    "\nclass ",
    "\nassert ",
    "\n#",
    "\nprint(",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a small MBPP subset.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-tasks", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def load_tasks(max_tasks: int):
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MBPP evaluation requires the optional 'data' dependencies. "
            "Install with: python -m pip install -e '.[data]'"
        ) from exc
    return load_dataset("mbpp", "sanitized", split=f"test[:{max_tasks}]")


def trim_completion(text: str) -> str:
    for marker in STOP_MARKERS:
        index = text.find(marker)
        if index > 0:
            text = text[:index]
    return text.rstrip()


def normalize_body_completion(text: str, indent: str = "    ") -> str:
    text = trim_completion(text).replace("\r\n", "\n")
    text = text.lstrip("\n")
    if not text.strip():
        return ""
    body = textwrap.dedent(text).lstrip()
    lines = body.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    normalized = "\n".join(f"{indent}{line}" if line.strip() else "" for line in lines)
    return normalized + ("\n" if normalized else "")


def extract_starter_code(reference_code: str) -> str:
    lines = reference_code.splitlines()
    starter_lines: list[str] = []
    inside_signature = False

    for line in lines:
        stripped = line.strip()
        if not inside_signature and (stripped.startswith("import ") or stripped.startswith("from ")):
            starter_lines.append(line)
            continue
        if not inside_signature and stripped.startswith("def "):
            inside_signature = True
            starter_lines.append(line)
            if stripped.endswith(":"):
                break
            continue
        if inside_signature:
            starter_lines.append(line)
            if stripped.endswith(":"):
                break

    starter = "\n".join(starter_lines).rstrip()
    if not starter:
        return ""
    return starter + "\n    "


def greedy_generate(
    model: torch.nn.Module,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> tuple[str, int, float]:
    token_ids = list(prompt.encode("utf-8"))
    generated = list(token_ids)
    segment_ids = [0] * len(token_ids)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)
            if hasattr(model, "segment_embeddings") and getattr(model, "segment_embeddings") is not None:
                segment_tensor = torch.tensor([segment_ids], dtype=torch.long, device=device)
                output = model(input_ids, segment_ids=segment_tensor)
            else:
                output = model(input_ids)
            next_token = int(torch.argmax(output.logits[0, -1]).item())
            generated.append(next_token)
            segment_ids.append(1)
            if next_token == ByteTokenizer.eos_token_id:
                break
    elapsed = time.perf_counter() - start
    completion = ByteTokenizer().decode(generated[len(token_ids) :])
    return trim_completion(completion), max(0, len(generated) - len(token_ids)), elapsed


def build_candidate_source(task_prompt: str, starter_code: str, completion: str) -> str:
    header = f"# Task: {task_prompt.strip()}\n"
    body = starter_code + normalize_body_completion(completion)
    if body and not body.endswith("\n"):
        body += "\n"
    return header + body


def run_mbpp_check(source: str, test_imports: list[str], test_list: list[str], timeout_sec: float) -> tuple[bool, str]:
    harness_parts = [source]
    if test_imports:
        harness_parts.extend(test_imports)
    harness_parts.extend(test_list)
    harness = "\n".join(harness_parts) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
        handle.write(harness)
        temp_path = Path(handle.name)
    try:
        completed = subprocess.run(
            [sys.executable, str(temp_path)],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        temp_path.unlink(missing_ok=True)

    if completed.returncode == 0:
        return True, "passed"
    stderr = completed.stderr.strip()
    stdout = completed.stdout.strip()
    message = stderr or stdout or f"failed(returncode={completed.returncode})"
    return False, message[:500]


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    payload = torch.load(args.checkpoint, map_location=device)
    config = ACEAtlasConfig.from_dict(payload["model_config"])
    model_name = payload["model_name"]
    model = build_model(model_name, config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    tasks = load_tasks(args.max_tasks)
    results = []
    generation_times = []
    generated_token_counts = []
    peak_memory_values = []

    for row in tasks:
        starter_code = extract_starter_code(row["code"])
        prompt = build_candidate_source(row["prompt"], starter_code, "")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        completion, generated_tokens, elapsed = greedy_generate(
            model=model,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        candidate_source = build_candidate_source(row["prompt"], starter_code, completion)
        passed, status = run_mbpp_check(
            source=candidate_source,
            test_imports=row.get("test_imports", []),
            test_list=row["test_list"],
            timeout_sec=args.timeout_sec,
        )
        results.append(
            {
                "task_id": row["task_id"],
                "passed": passed,
                "status": status,
                "starter_code": starter_code,
                "completion_preview": completion[:300],
                "generated_tokens": generated_tokens,
                "generation_time_sec": elapsed,
            }
        )
        generation_times.append(elapsed)
        generated_token_counts.append(generated_tokens)
        if device.type == "cuda":
            peak_memory_values.append(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    passed_count = sum(1 for row in results if row["passed"])
    total_generated = sum(generated_token_counts)
    total_time = sum(generation_times)
    metrics = {
        "model_name": model_name,
        "parameter_count": count_parameters(model)["total"],
        "parameter_count_human": format_parameter_count(count_parameters(model)["total"]),
        "checkpoint": str(args.checkpoint),
        "benchmark": "mbpp_sanitized",
        "protocol": {
            "subset": f"first_{args.max_tasks}_tasks",
            "prompt_style": "task_text_plus_function_signature",
            "decoding": "greedy",
            "max_new_tokens": args.max_new_tokens,
            "timeout_sec": args.timeout_sec,
        },
        "tasks": len(results),
        "passed": passed_count,
        "pass_at_1": passed_count / len(results) if results else 0.0,
        "avg_generation_time_sec": total_time / len(generation_times) if generation_times else 0.0,
        "avg_generated_tokens": total_generated / len(generated_token_counts) if generated_token_counts else 0.0,
        "avg_generated_tokens_per_sec": total_generated / total_time if total_time > 0 else 0.0,
        "max_peak_memory_mb": max(peak_memory_values) if peak_memory_values else None,
        "results": results,
    }

    print(json.dumps(metrics, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
