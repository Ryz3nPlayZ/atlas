from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ace_atlas.benchmarks.recall import generate_mqar_cases, score_recall_cases


def main() -> None:
    cases = generate_mqar_cases(num_cases=5, num_pairs=6)
    predictions = [case.expected for case in cases]
    score = score_recall_cases(cases, predictions)
    print(f"MQAR adapter smoke score: {score.correct}/{score.total} = {score.accuracy:.3f}")
    print("Sample prompt:")
    print(cases[0].prompt)


if __name__ == "__main__":
    main()

