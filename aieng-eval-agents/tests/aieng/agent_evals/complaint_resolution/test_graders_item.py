"""Tests for complaint_resolution graders.item — item_level_deterministic_grader."""

import pytest
from aieng.agent_evals.complaint_resolution.graders.item import item_level_deterministic_grader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(
    predicted_category: str | None,
    cited_policy_ids: str,
    expected_category: str,
    gold_policy_id: str,
) -> dict[str, float]:
    """Run the grader and return {name: value} for easy assertions."""
    output = {
        "predicted_category": predicted_category,
        "cited_policy_ids": cited_policy_ids,
    }
    expected_output = {
        "category": expected_category,
        "gold_policy_id": gold_policy_id,
    }
    evals = item_level_deterministic_grader(
        input="some narrative",
        output=output,
        expected_output=expected_output,
    )
    return {e.name: e.value for e in evals}


# ---------------------------------------------------------------------------
# Return shape
# ---------------------------------------------------------------------------


class TestReturnShape:
    """The grader always emits exactly four Evaluation objects."""

    def test_returns_four_evaluations(self):
        """Grader emits exactly four Evaluation objects."""
        evals = item_level_deterministic_grader(
            input="x",
            output={"predicted_category": "credit_card", "cited_policy_ids": "POL-CREDIT-CARD"},
            expected_output={"category": "credit_card", "gold_policy_id": "POL-CREDIT-CARD"},
        )
        assert len(evals) == 4

    def test_returns_expected_metric_names(self):
        """Grader emits the four expected metric names."""
        evals = item_level_deterministic_grader(
            input="x",
            output={"predicted_category": "credit_card", "cited_policy_ids": "POL-CREDIT-CARD"},
            expected_output={"category": "credit_card", "gold_policy_id": "POL-CREDIT-CARD"},
        )
        names = {e.name for e in evals}
        assert names == {"category_correct", "policy_match", "retrieval_precision", "retrieval_coverage"}


# ---------------------------------------------------------------------------
# category_correct
# ---------------------------------------------------------------------------


class TestCategoryCorrect:
    """Tests for the category_correct metric."""

    def test_correct_classification(self):
        """category_correct is 1.0 when categories match."""
        metrics = _run("credit_card", "POL-CREDIT-CARD", "credit_card", "POL-CREDIT-CARD")
        assert metrics["category_correct"] == 1.0

    def test_wrong_classification(self):
        """category_correct is 0.0 when categories do not match."""
        metrics = _run("debt_collection", "POL-DEBT-COLLECTION", "credit_card", "POL-CREDIT-CARD")
        assert metrics["category_correct"] == 0.0

    def test_case_insensitive_match(self):
        """category_correct is 1.0 when categories match modulo case."""
        metrics = _run("CREDIT_CARD", "POL-CREDIT-CARD", "credit_card", "POL-CREDIT-CARD")
        assert metrics["category_correct"] == 1.0

    def test_none_prediction_is_wrong(self):
        """category_correct is 0.0 when predicted_category is None."""
        metrics = _run(None, "", "credit_card", "POL-CREDIT-CARD")
        assert metrics["category_correct"] == 0.0

    def test_all_five_canonical_categories(self):
        """category_correct is 1.0 for each of the five canonical category labels."""
        for cat, pol in [
            ("credit_card", "POL-CREDIT-CARD"),
            ("credit_reporting", "POL-CREDIT-REPORTING"),
            ("debt_collection", "POL-DEBT-COLLECTION"),
            ("mortgages_and_loans", "POL-MORTGAGE-LOANS"),
            ("retail_banking", "POL-RETAIL-BANKING"),
        ]:
            metrics = _run(cat, pol, cat, pol)
            assert metrics["category_correct"] == 1.0, f"failed for {cat}"


# ---------------------------------------------------------------------------
# policy_match
# ---------------------------------------------------------------------------


class TestPolicyMatch:
    """Tests for the policy_match metric."""

    def test_exact_match(self):
        """policy_match is 1.0 when the gold id is cited."""
        metrics = _run("credit_card", "POL-CREDIT-CARD", "credit_card", "POL-CREDIT-CARD")
        assert metrics["policy_match"] == 1.0

    def test_no_match(self):
        """policy_match is 0.0 when the gold id is not in the cited set."""
        metrics = _run("credit_card", "POL-RETAIL-BANKING", "credit_card", "POL-CREDIT-CARD")
        assert metrics["policy_match"] == 0.0

    def test_gold_id_among_multiple_cited(self):
        """policy_match is 1.0 when gold id appears among multiple cited ids."""
        metrics = _run(
            "credit_card",
            "POL-CREDIT-CARD, POL-RETAIL-BANKING",
            "credit_card",
            "POL-CREDIT-CARD",
        )
        assert metrics["policy_match"] == 1.0

    def test_empty_cited_is_no_match(self):
        """policy_match is 0.0 when cited_policy_ids is empty."""
        metrics = _run("credit_card", "", "credit_card", "POL-CREDIT-CARD")
        assert metrics["policy_match"] == 0.0

    def test_case_insensitive_id_match(self):
        """policy_match is 1.0 when ids match modulo case."""
        metrics = _run("credit_card", "pol-credit-card", "credit_card", "POL-CREDIT-CARD")
        assert metrics["policy_match"] == 1.0


# ---------------------------------------------------------------------------
# retrieval_precision
# ---------------------------------------------------------------------------


class TestRetrievalPrecision:
    """Tests for the retrieval_precision metric."""

    def test_single_correct_id_precision_is_one(self):
        """Precision is 1.0 when the single cited id is correct."""
        metrics = _run("credit_card", "POL-CREDIT-CARD", "credit_card", "POL-CREDIT-CARD")
        assert metrics["retrieval_precision"] == pytest.approx(1.0)

    def test_one_correct_one_wrong_precision_is_half(self):
        """Precision is 0.5 when one of two cited ids is the gold id."""
        metrics = _run(
            "credit_card",
            "POL-CREDIT-CARD, POL-RETAIL-BANKING",
            "credit_card",
            "POL-CREDIT-CARD",
        )
        assert metrics["retrieval_precision"] == pytest.approx(0.5)

    def test_all_wrong_precision_is_zero(self):
        """Precision is 0.0 when no cited id is gold."""
        metrics = _run("credit_card", "POL-RETAIL-BANKING", "credit_card", "POL-CREDIT-CARD")
        assert metrics["retrieval_precision"] == pytest.approx(0.0)

    def test_empty_cited_precision_is_zero(self):
        """Precision is 0.0 when no ids are cited (avoids zero-division)."""
        metrics = _run("credit_card", "", "credit_card", "POL-CREDIT-CARD")
        assert metrics["retrieval_precision"] == pytest.approx(0.0)

    def test_three_cited_one_correct_precision_is_one_third(self):
        """Precision is 1/3 when one of three cited ids is the gold id."""
        metrics = _run(
            "credit_card",
            "POL-CREDIT-CARD, POL-RETAIL-BANKING, POL-DEBT-COLLECTION",
            "credit_card",
            "POL-CREDIT-CARD",
        )
        assert metrics["retrieval_precision"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# retrieval_coverage
# ---------------------------------------------------------------------------


class TestRetrievalCoverage:
    """Tests for the retrieval_coverage metric."""

    def test_gold_id_cited_coverage_is_one(self):
        """Coverage is 1.0 when the single gold id is cited."""
        metrics = _run("credit_card", "POL-CREDIT-CARD", "credit_card", "POL-CREDIT-CARD")
        assert metrics["retrieval_coverage"] == pytest.approx(1.0)

    def test_gold_id_not_cited_coverage_is_zero(self):
        """Coverage is 0.0 when the gold id is absent from cited ids."""
        metrics = _run("credit_card", "POL-RETAIL-BANKING", "credit_card", "POL-CREDIT-CARD")
        assert metrics["retrieval_coverage"] == pytest.approx(0.0)

    def test_empty_cited_coverage_is_zero(self):
        """Coverage is 0.0 when no ids are cited."""
        metrics = _run("credit_card", "", "credit_card", "POL-CREDIT-CARD")
        assert metrics["retrieval_coverage"] == pytest.approx(0.0)

    def test_coverage_independent_of_extra_cited_ids(self):
        """Coverage remains 1.0 regardless of extra (wrong) cited ids."""
        metrics = _run(
            "credit_card",
            "POL-CREDIT-CARD, POL-RETAIL-BANKING, POL-DEBT-COLLECTION",
            "credit_card",
            "POL-CREDIT-CARD",
        )
        assert metrics["retrieval_coverage"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# metadata fields
# ---------------------------------------------------------------------------


class TestMetadata:
    """The grader attaches meaningful metadata to each Evaluation."""

    def _evals_by_name(self, *args, **kwargs):
        evals = item_level_deterministic_grader(*args, **kwargs)
        return {e.name: e for e in evals}

    def test_category_correct_metadata_contains_expected_and_actual(self):
        """category_correct metadata exposes expected and actual labels."""
        by_name = self._evals_by_name(
            input="x",
            output={"predicted_category": "credit_card", "cited_policy_ids": "POL-CREDIT-CARD"},
            expected_output={"category": "retail_banking", "gold_policy_id": "POL-RETAIL-BANKING"},
        )
        meta = by_name["category_correct"].metadata
        assert meta is not None
        assert "expected" in meta
        assert "actual" in meta

    def test_policy_match_metadata_contains_gold_and_predicted(self):
        """policy_match metadata exposes gold and predicted id lists."""
        by_name = self._evals_by_name(
            input="x",
            output={"predicted_category": "credit_card", "cited_policy_ids": "POL-CREDIT-CARD"},
            expected_output={"category": "credit_card", "gold_policy_id": "POL-CREDIT-CARD"},
        )
        meta = by_name["policy_match"].metadata
        assert meta is not None
        assert "gold" in meta
        assert "predicted" in meta

    def test_extra_kwargs_ignored(self):
        """Extra keyword arguments are silently ignored (interface compatibility)."""
        evals = item_level_deterministic_grader(
            input="x",
            output={"predicted_category": "credit_card", "cited_policy_ids": "POL-CREDIT-CARD"},
            expected_output={"category": "credit_card", "gold_policy_id": "POL-CREDIT-CARD"},
            metadata={"foo": "bar"},
            extra_unused_kwarg=42,
        )
        assert len(evals) == 4
