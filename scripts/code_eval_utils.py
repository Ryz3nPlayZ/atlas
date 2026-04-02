from __future__ import annotations

import ast
import textwrap


STOP_MARKERS = (
    "\ndef ",
    "\nclass ",
    "\nif __name__",
    "\nassert ",
    "\n#",
    "\nprint(",
)


def trim_completion(text: str, stop_markers: tuple[str, ...] = STOP_MARKERS) -> str:
    for marker in stop_markers:
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


def longest_parsable_body(prompt_source: str, completion: str, indent: str = "    ") -> str:
    prompt_source = prompt_source.rstrip() + "\n"
    normalized = normalize_body_completion(completion, indent=indent)
    if not normalized:
        return normalized

    lines = normalized.splitlines()
    while lines:
        candidate = "\n".join(lines)
        if candidate and not candidate.endswith("\n"):
            candidate += "\n"
        try:
            ast.parse(prompt_source + candidate)
            return candidate
        except SyntaxError:
            lines.pop()

    return ""


def repair_body_completion(prompt_source: str, completion: str, indent: str = "    ") -> str:
    prompt_source = prompt_source.rstrip() + "\n"
    candidate = longest_parsable_body(prompt_source, completion, indent=indent)
    if candidate:
        return candidate

    normalized = normalize_body_completion(completion, indent=indent)
    if not normalized:
        return normalized

    stripped = normalized.rstrip()
    if stripped.endswith(":"):
        fallback = stripped + "\n" + (indent * 2) + "pass\n"
        try:
            ast.parse(prompt_source + fallback)
            return fallback
        except SyntaxError:
            return ""

    return ""
