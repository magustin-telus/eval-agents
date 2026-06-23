"""Tests for complaint_resolution graders.run — run_level_grader."""

from types import SimpleNamespace

import pytest
from aieng.agent_evals.complaint_resolution.graders.run import run_level_grader
from aieng.agent_evals.complaint_resolution.kb import CATEGORY_LABELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item_result(predicted: str | None, expected: str | None) -> object:
    """Build a minimal ExperimentItemResult-like object for run_level_grader."""
    return SimpleNamespace(
        item={"expected_output": {"category": expected} if expected is not None else {}},
        output={"predicted_category": predicted} if predicted is not None else {},
    )


def _run(pairs: list[tuple[str | None, str | None]]) -> dict[str, float]:
    """Run the grader over a list of (predicted, expected) pairs and return metric dict."""
    item_results = [_make_item_result(pred, exp) for pred, exp in pairs]
    evals = run_level_grader(item_results=item_results)
    return {e.name: e.value for e in evals}


# ---------------------------------------------------------------------------
# Return shape
# ---------------------------------------------------------------------------


class TestReturnShape:
    """The grader always emits exactly five Evaluation objects."""

    def test_returns_five_evaluations(self):
        """run_level_grader emits exactly five Evaluation objects."""
        evals = run_level_grader(
            item_results=[_make_item_result("credit_card", "credit_card")]
        )
        assert len(evals) == 5

    def test_returns_expected_metric_names(self):
        """run_level_grader emits the five expected metric names."""
        evals = run_level_grader(
            item_results=[_make_item_result("credit_card", "credit_card")]
        )
        names = {e.name for e in evals}
        assert names == {
            "category_macro_precision",
            "category_macro_recall",
            "category_macro_f1",
            "category_accuracy",
            "category_confusion_matrix",
        }


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    """Grader handles an empty item_results list gracefully."""

    def test_empty_item_results_all_zero(self):
        """All metrics are 0.0 when item_results is empty."""
        metrics = _run([])
        assert metrics["category_macro_f1"] == pytest.approx(0.0)
        assert metrics["category_accuracy"] == pytest.approx(0.0)
        assert metrics["category_macro_precision"] == pytest.approx(0.0)
        assert metrics["category_macro_recall"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Perfect predictions
# ---------------------------------------------------------------------------


class TestPerfectPredictions:
    """Grader returns 1.0 for all metrics when every prediction is correct."""

    def test_all_correct_gives_one(self):
        """Accuracy and macro-F1 are both 1.0 when all predictions are correct."""
        pairs = [(cat, cat) for cat in CATEGORY_LABELS]
        metrics = _run(pairs)
        assert metrics["category_accuracy"] == pytest.approx(1.0)
        assert metrics["category_macro_f1"] == pytest.approx(1.0)
        assert metrics["category_macro_precision"] == pytest.approx(1.0)
        assert metrics["category_macro_recall"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# All wrong predictions
# ---------------------------------------------------------------------------


class TestAllWrongPredictions:
    """Grader returns 0.0 accuracy when no prediction is correct."""

    def test_all_wrong_accuracy_is_zero(self):
        """Accuracy is 0.0 when every predicted category is wrong."""
        pairs = [("retail_banking", "credit_card"), ("credit_card", "retail_banking")]
        metrics = _run(pairs)
        assert metrics["category_accuracy"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Mixed predictions
# ---------------------------------------------------------------------------


class TestMixedPredictions:
    """Grader computes correct partial-accuracy values for mixed results."""

    def test_half_correct_accuracy(self):
        """Accuracy is 0.5 when half the predictions are correct."""
        pairs = [
            ("credit_card", "credit_card"),
            ("retail_banking", "credit_card"),
        ]
        metrics = _run(pairs)
        assert metrics["category_accuracy"] == pytest.approx(0.5)

    def test_three_quarters_correct_accuracy(self):
        """Accuracy is 0.75 when three of four predictions are correct."""
        pairs = [
            ("credit_card", "credit_card"),
            ("credit_card", "credit_card"),
            ("credit_card", "credit_card"),
            ("retail_banking", "credit_card"),
        ]
        metrics = _run(pairs)
        assert metrics["category_accuracy"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# INVALID bucket
# ---------------------------------------------------------------------------


class TestInvalidBucket:
    """None / unknown predictions are bucketed as INVALID without raising."""

    def test_none_prediction_bucketed_as_invalid(self):
        """None predicted_category is treated as INVALID and does not crash."""
        metrics = _run([(None, "credit_card")])
        # Accuracy must be 0 (INVALID ≠ credit_card)
        assert metrics["category_accuracy"] == pytest.approx(0.0)

    def test_unknown_category_bucketed_as_invalid(self):
        """An out-of-vocabulary predicted category is treated as INVALID."""
        metrics = _run([("completely_unknown_category", "credit_card")])
        assert metrics["category_accuracy"] == pytest.approx(0.0)

    def test_invalid_expected_category_is_skipped(self):
        """Items with an invalid/None expected category are skipped."""
        # Only the second item has a valid expected category.
        item_results = [
            _make_item_result("credit_card", None),
            _make_item_result("retail_banking", "retail_banking"),
        ]
        evals = run_level_grader(item_results=item_results)
        metrics = {e.name: e.value for e in evals}
        # Only the valid item contributes — accuracy should be 1.0
        assert metrics["category_accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


class TestConfusionMatrix:
    """Grader includes a confusion matrix in the category_confusion_matrix metadata."""

    def test_confusion_matrix_present_in_metadata(self):
        """Metadata for category_confusion_matrix contains 'matrix' and 'labels'."""
        evals = run_level_grader(
            item_results=[_make_item_result("credit_card", "credit_card")]
        )
        cm_eval = next(e for e in evals if e.name == "category_confusion_matrix")
        assert cm_eval.metadata is not None
        assert "matrix" in cm_eval.metadata
        assert "labels" in cm_eval.metadata

    def test_confusion_matrix_is_square(self):
        """Confusion matrix dimensions match the number of labels."""
        evals = run_level_grader(
            item_results=[_make_item_result("credit_card", "credit_card")]
        )
        cm_eval = next(e for e in evals if e.name == "category_confusion_matrix")
        labels = cm_eval.metadata["labels"]
        matrix = cm_eval.metadata["matrix"]
        n = len(labels)
        assert len(matrix) == n
        assert all(len(row) == n for row in matrix)

    def test_single_correct_prediction_diagonal(self):
        """A single correct prediction has a 1 on the diagonal of the matrix."""
        evals = run_level_grader(
            item_results=[_make_item_result("credit_card", "credit_card")]
        )
        cm_eval = next(e for e in evals if e.name == "category_confusion_matrix")
        labels: list[str] = cm_eval.metadata["labels"]
        matrix: list[list[int]] = cm_eval.metadata["matrix"]
        idx = labels.index("credit_card")
        assert matrix[idx][idx] == 1

    def test_misclassified_prediction_off_diagonal(self):
        """A wrong prediction appears off-diagonal."""
        evals = run_level_grader(
            item_results=[_make_item_result("retail_banking", "credit_card")]
        )
        cm_eval = next(e for e in evals if e.name == "category_confusion_matrix")
        labels: list[str] = cm_eval.metadata["labels"]
        matrix: list[list[int]] = cm_eval.metadata["matrix"]
        true_idx = labels.index("credit_card")
        pred_idx = labels.index("retail_banking")
        assert matrix[true_idx][pred_idx] == 1

    def test_confusion_matrix_value_is_item_count(self):
        """The value of category_confusion_matrix equals the number of valid items."""
        item_results = [_make_item_result(cat, cat) for cat in CATEGORY_LABELS]
        evals = run_level_grader(item_results=item_results)
        cm_eval = next(e for e in evals if e.name == "category_confusion_matrix")
        assert cm_eval.value == pytest.approx(float(len(CATEGORY_LABELS)))


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    """Grader attaches consistent metadata to all five metrics."""

    def test_valid_items_count_in_metadata(self):
        """Metadata carries a 'valid_items' count equal to the number of valid items."""
        pairs = [(cat, cat) for cat in CATEGORY_LABELS]
        item_results = [_make_item_result(pred, exp) for pred, exp in pairs]
        evals = run_level_grader(item_results=item_results)
        for e in evals:
            if e.metadata and "valid_items" in e.metadata:
                assert e.metadata["valid_items"] == len(CATEGORY_LABELS)
                break
        else:
            pytest.fail("No evaluation had valid_items in metadata")

    def test_extra_kwargs_ignored(self):
        """Extra keyword arguments are silently ignored."""
        evals = run_level_grader(
            item_results=[_make_item_result("credit_card", "credit_card")],
            ignored_kwarg="should_not_crash",
        )
        assert len(evals) == 5
