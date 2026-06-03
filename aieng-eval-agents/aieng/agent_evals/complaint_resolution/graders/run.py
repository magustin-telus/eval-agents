"""Run-level graders for complaint-resolution evaluation.

Aggregates item-level classification predictions into run-level multiclass metrics
(precision/recall/F1, macro-F1, and a confusion matrix) over the complaint categories.
"""

from typing import Any

from aieng.agent_evals.evaluation import Evaluation, ExperimentItemResult
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from ._common import CATEGORY_LABELS, extract_expected_output, get_field, normalize_category


def run_level_grader(*, item_results: list[ExperimentItemResult], **kwargs: Any) -> list[Evaluation]:
    """Compute run-level classification metrics over experiment item results.

    Parameters
    ----------
    item_results : list[ExperimentItemResult]
        Item results emitted by a Langfuse experiment run.
    **kwargs : Any
        Additional run-evaluator kwargs. Ignored.

    Returns
    -------
    list[Evaluation]
        Run-level metrics:

        - ``category_macro_precision``
        - ``category_macro_recall``
        - ``category_macro_f1``
        - ``category_accuracy``
        - ``category_confusion_matrix`` (labels + matrix in metadata)
    """
    del kwargs  # Unused but part of the evaluator interface.

    expected: list[str] = []
    predicted: list[str] = []
    invalid_expected = 0
    invalid_predicted = 0

    for item_result in item_results:
        expected_output = extract_expected_output(item_result)
        predicted_output = item_result.output

        expected_category = normalize_category(get_field(expected_output, "category"))
        predicted_category = normalize_category(get_field(predicted_output, "predicted_category"))

        # Skip items with no ground-truth label; bucket invalid predictions as INVALID.
        if expected_category is None:
            continue
        if expected_category not in CATEGORY_LABELS:
            invalid_expected += 1
            expected_category = "INVALID"
        if predicted_category is None or predicted_category not in CATEGORY_LABELS:
            invalid_predicted += 1
            predicted_category = "INVALID"

        expected.append(expected_category)
        predicted.append(predicted_category)

    labels = list(CATEGORY_LABELS) + (["INVALID"] if invalid_expected or invalid_predicted else [])

    macro_precision = macro_recall = macro_f1 = accuracy = 0.0
    matrix = [[0 for _ in labels] for _ in labels]
    if expected:
        macro_precision_v, macro_recall_v, macro_f1_v, _ = precision_recall_fscore_support(
            expected, predicted, labels=labels, average="macro", zero_division=0
        )
        macro_precision = float(macro_precision_v)
        macro_recall = float(macro_recall_v)
        macro_f1 = float(f1_score(expected, predicted, labels=labels, average="macro", zero_division=0))
        accuracy = sum(e == p for e, p in zip(expected, predicted, strict=True)) / len(expected)
        matrix = confusion_matrix(expected, predicted, labels=labels).tolist()

    common_meta = {
        "labels": labels,
        "valid_items": len(expected),
        "invalid_expected_count": invalid_expected,
        "invalid_predicted_count": invalid_predicted,
    }

    return [
        Evaluation(name="category_macro_precision", value=macro_precision, metadata=common_meta),
        Evaluation(name="category_macro_recall", value=macro_recall, metadata=common_meta),
        Evaluation(name="category_macro_f1", value=macro_f1, metadata=common_meta),
        Evaluation(name="category_accuracy", value=accuracy, metadata=common_meta),
        Evaluation(
            name="category_confusion_matrix",
            value=float(len(expected)),
            metadata={**common_meta, "matrix": matrix},
        ),
    ]


__all__ = ["run_level_grader"]
