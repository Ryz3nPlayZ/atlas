from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ace_atlas.benchmarks.verifier import CodeCase, MathCase, verify_code_cases, verify_math_cases


def main() -> None:
    code_cases = [
        CodeCase(
            prompt="Print the sum of 2 and 3.",
            generated_code="print(2 + 3)",
            expected_stdout="5",
        )
    ]
    math_cases = [
        MathCase(
            prompt="What is 7 * (4 + 2)?",
            generated_expression="7 * (4 + 2)",
            expected_value="42",
        )
    ]
    code_score = verify_code_cases(code_cases)
    math_score = verify_math_cases(math_cases)
    print(f"Code verifier: {code_score.passed}/{code_score.total} = {code_score.accuracy:.3f}")
    print(f"Math verifier: {math_score.passed}/{math_score.total} = {math_score.accuracy:.3f}")


if __name__ == "__main__":
    main()

