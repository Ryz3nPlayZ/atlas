from ace_atlas.benchmarks.recall import generate_mqar_cases, score_recall_cases
from ace_atlas.benchmarks.verifier import CodeCase, MathCase, verify_code_cases, verify_math_cases


def test_recall_adapter_scores_perfect_predictions() -> None:
    cases = generate_mqar_cases(num_cases=3, num_pairs=4, seed=3)
    predictions = [case.expected for case in cases]
    score = score_recall_cases(cases, predictions)
    assert score.total == 3
    assert score.correct == 3


def test_verifier_adapters_accept_valid_outputs() -> None:
    code_score = verify_code_cases(
        [
            CodeCase(
                prompt="Print 9.",
                generated_code="print(9)",
                expected_stdout="9",
            )
        ]
    )
    math_score = verify_math_cases(
        [
            MathCase(
                prompt="Compute 6 * 7.",
                generated_expression="6 * 7",
                expected_value="42",
            )
        ]
    )
    assert code_score.passed == 1
    assert math_score.passed == 1

