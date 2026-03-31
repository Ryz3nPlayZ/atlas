from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(slots=True)
class MQARCase:
    prompt: str
    expected: str


@dataclass(slots=True)
class RecallScore:
    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


def generate_mqar_cases(
    num_cases: int,
    num_pairs: int = 8,
    seed: int = 7,
) -> list[MQARCase]:
    rng = random.Random(seed)
    cases: list[MQARCase] = []
    for _ in range(num_cases):
        keys = [f"K{idx}" for idx in range(num_pairs)]
        vals = [f"V{rng.randint(100, 999)}" for _ in range(num_pairs)]
        pairs = list(zip(keys, vals))
        rng.shuffle(pairs)
        query_key, query_value = rng.choice(pairs)
        serialized_pairs = " ".join(f"{k}:{v}" for k, v in pairs)
        prompt = (
            "Memorize these pairs and answer with the value only.\n"
            f"{serialized_pairs}\n"
            f"Question: what is the value for {query_key}?"
        )
        cases.append(MQARCase(prompt=prompt, expected=query_value))
    return cases


def score_recall_cases(cases: list[MQARCase], predictions: list[str]) -> RecallScore:
    if len(cases) != len(predictions):
        raise ValueError("cases and predictions must have the same length")
    correct = 0
    for case, prediction in zip(cases, predictions):
        if prediction.strip() == case.expected:
            correct += 1
    return RecallScore(total=len(cases), correct=correct)

