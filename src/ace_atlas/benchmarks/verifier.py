from __future__ import annotations

from dataclasses import dataclass
import ast

from ace_atlas.tools.python_executor import ExecutionResult, PythonExecutor


@dataclass(slots=True)
class CodeCase:
    prompt: str
    generated_code: str
    expected_stdout: str


@dataclass(slots=True)
class MathCase:
    prompt: str
    generated_expression: str
    expected_value: str


@dataclass(slots=True)
class VerifierScore:
    total: int
    passed: int

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total


def verify_code_cases(cases: list[CodeCase], executor: PythonExecutor | None = None) -> VerifierScore:
    executor = executor or PythonExecutor()
    passed = 0
    for case in cases:
        result = executor.execute(case.generated_code)
        if not result.timed_out and result.returncode == 0 and result.stdout.strip() == case.expected_stdout.strip():
            passed += 1
    return VerifierScore(total=len(cases), passed=passed)


def _safe_eval_expression(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
        ast.Load,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f"Disallowed node in math expression: {type(node).__name__}")
    return str(eval(compile(tree, "<math_case>", "eval"), {"__builtins__": {}}, {}))


def verify_math_cases(cases: list[MathCase]) -> VerifierScore:
    passed = 0
    for case in cases:
        try:
            value = _safe_eval_expression(case.generated_expression)
        except Exception:
            continue
        if value == case.expected_value:
            passed += 1
    return VerifierScore(total=len(cases), passed=passed)

