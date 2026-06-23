"""Tests for complaint_resolution.data_analysis utility functions.

All tests use a small in-memory fixture dataset so no Kaggle network calls
are made and no embedding endpoint is needed.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aieng.agent_evals.complaint_resolution.data_analysis import (
    AgentTestResult,
    CategoryStats,
    RetrievalResult,
    compute_agent_metrics,
    compute_retrieval_metrics,
)
from aieng.agent_evals.complaint_resolution.kb import CATEGORY_LABELS


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_retrieval_results(category: str, found: bool, total: int = 3) -> list[RetrievalResult]:
    """Build a list of RetrievalResult objects for one category."""
    results = []
    gold_id = f"POL-{category.upper().replace('_', '-')}"
    for i in range(total):
        retrieved_ids = [gold_id] if found else ["POL-WRONG-POLICY"]
        results.append(
            RetrievalResult(
                example_id=i,
                category=category,
                gold_policy_id=gold_id,
                retrieved_ids=retrieved_ids,
                gold_rank=1 if found else None,
            )
        )
    return results


# ---------------------------------------------------------------------------
# CategoryStats dataclass
# ---------------------------------------------------------------------------


class TestCategoryStats:
    """Tests for the CategoryStats dataclass."""

    def test_construction(self):
        """CategoryStats stores all six fields."""
        stats = CategoryStats(
            name="credit_card",
            count=100,
            percentage=25.0,
            avg_narrative_length=50.5,
            min_narrative_length=10,
            max_narrative_length=200,
        )
        assert stats.name == "credit_card"
        assert stats.count == 100
        assert stats.percentage == pytest.approx(25.0)
        assert stats.avg_narrative_length == pytest.approx(50.5)
        assert stats.min_narrative_length == 10
        assert stats.max_narrative_length == 200


# ---------------------------------------------------------------------------
# RetrievalResult dataclass
# ---------------------------------------------------------------------------


class TestRetrievalResult:
    """Tests for the RetrievalResult dataclass."""

    def test_construction_with_rank(self):
        """RetrievalResult stores all fields including gold_rank."""
        r = RetrievalResult(
            example_id=0,
            category="credit_card",
            gold_policy_id="POL-CREDIT-CARD",
            retrieved_ids=["POL-CREDIT-CARD", "POL-RETAIL-BANKING"],
            gold_rank=1,
        )
        assert r.gold_rank == 1
        assert r.retrieved_ids == ["POL-CREDIT-CARD", "POL-RETAIL-BANKING"]

    def test_construction_gold_not_found(self):
        """RetrievalResult gold_rank is None when gold not retrieved."""
        r = RetrievalResult(
            example_id=1,
            category="credit_card",
            gold_policy_id="POL-CREDIT-CARD",
            retrieved_ids=["POL-RETAIL-BANKING"],
            gold_rank=None,
        )
        assert r.gold_rank is None


# ---------------------------------------------------------------------------
# AgentTestResult dataclass
# ---------------------------------------------------------------------------


class TestAgentTestResult:
    """Tests for the AgentTestResult dataclass."""

    def test_successful_result(self):
        """Successful AgentTestResult has success=True and no error."""
        r = AgentTestResult(
            example_id=0,
            gold_category="credit_card",
            predicted_category="credit_card",
            cited_policy_ids="POL-CREDIT-CARD",
            success=True,
        )
        assert r.success is True
        assert r.error is None

    def test_failed_result(self):
        """Failed AgentTestResult has success=False and an error message."""
        r = AgentTestResult(
            example_id=1,
            gold_category="credit_card",
            predicted_category=None,
            cited_policy_ids="",
            success=False,
            error="Timeout",
        )
        assert r.success is False
        assert r.error == "Timeout"


# ---------------------------------------------------------------------------
# compute_retrieval_metrics
# ---------------------------------------------------------------------------


class TestComputeRetrievalMetrics:
    """Tests for compute_retrieval_metrics — pure function over RetrievalResult lists."""

    def test_perfect_retrieval_precision_at_3_is_one(self):
        """precision_at_3 is 1.0 when gold is always retrieved."""
        results = {"credit_card": _make_retrieval_results("credit_card", found=True, total=3)}
        metrics = compute_retrieval_metrics(results)
        assert metrics["credit_card"]["precision_at_3"] == pytest.approx(1.0)

    def test_no_retrieval_precision_at_3_is_zero(self):
        """precision_at_3 is 0.0 when gold is never retrieved."""
        results = {"credit_card": _make_retrieval_results("credit_card", found=False, total=3)}
        metrics = compute_retrieval_metrics(results)
        assert metrics["credit_card"]["precision_at_3"] == pytest.approx(0.0)

    def test_partial_retrieval(self):
        """precision_at_3 is 0.5 when gold is found in 1 of 2 examples."""
        category = "credit_card"
        gold_id = "POL-CREDIT-CARD"
        results = {
            category: [
                RetrievalResult(
                    example_id=0,
                    category=category,
                    gold_policy_id=gold_id,
                    retrieved_ids=[gold_id],
                    gold_rank=1,
                ),
                RetrievalResult(
                    example_id=1,
                    category=category,
                    gold_policy_id=gold_id,
                    retrieved_ids=["POL-WRONG"],
                    gold_rank=None,
                ),
            ]
        }
        metrics = compute_retrieval_metrics(results)
        assert metrics[category]["precision_at_3"] == pytest.approx(0.5)

    def test_found_count_and_total_correct(self):
        """found_count and total reflect the fixture data."""
        results = {"retail_banking": _make_retrieval_results("retail_banking", found=True, total=4)}
        metrics = compute_retrieval_metrics(results)
        assert metrics["retail_banking"]["found_count"] == 4
        assert metrics["retail_banking"]["total"] == 4

    def test_avg_rank_is_one_when_always_top_result(self):
        """avg_rank is 1.0 when the gold document is always rank 1."""
        results = {"credit_card": _make_retrieval_results("credit_card", found=True, total=2)}
        metrics = compute_retrieval_metrics(results)
        assert metrics["credit_card"]["avg_rank"] == pytest.approx(1.0)

    def test_avg_rank_is_none_when_never_found(self):
        """avg_rank is None when the gold document is never retrieved."""
        results = {"credit_card": _make_retrieval_results("credit_card", found=False, total=2)}
        metrics = compute_retrieval_metrics(results)
        assert metrics["credit_card"]["avg_rank"] is None

    def test_empty_results_list_gives_zero_precision(self):
        """An empty result list for a category gives 0 precision."""
        metrics = compute_retrieval_metrics({"credit_card": []})
        assert metrics["credit_card"]["precision_at_3"] == pytest.approx(0.0)

    def test_multiple_categories_all_present(self):
        """All categories in input appear as keys in the output metrics dict."""
        results = {cat: _make_retrieval_results(cat, found=True) for cat in CATEGORY_LABELS}
        metrics = compute_retrieval_metrics(results)
        assert set(metrics.keys()) == set(CATEGORY_LABELS)


# ---------------------------------------------------------------------------
# compute_agent_metrics
# ---------------------------------------------------------------------------


class TestComputeAgentMetrics:
    """Tests for compute_agent_metrics — pure function over AgentTestResult lists."""

    def _make_result(
        self,
        gold: str,
        predicted: str | None,
        success: bool,
        error: str | None = None,
    ) -> AgentTestResult:
        return AgentTestResult(
            example_id=0,
            gold_category=gold,
            predicted_category=predicted,
            cited_policy_ids="",
            success=success,
            error=error,
        )

    def test_empty_results(self):
        """Empty results list returns zero for all metrics."""
        m = compute_agent_metrics([])
        assert m["success_rate"] == 0
        assert m["category_accuracy"] == 0

    def test_all_successful_correct(self):
        """success_rate and category_accuracy are 1.0 when all succeed and match."""
        results = [self._make_result("credit_card", "credit_card", success=True) for _ in range(5)]
        m = compute_agent_metrics(results)
        assert m["success_rate"] == pytest.approx(1.0)
        assert m["category_accuracy"] == pytest.approx(1.0)

    def test_all_failed(self):
        """success_rate is 0.0 when all results are failures."""
        results = [
            self._make_result("credit_card", None, success=False, error="timeout") for _ in range(3)
        ]
        m = compute_agent_metrics(results)
        assert m["success_rate"] == pytest.approx(0.0)

    def test_half_successful(self):
        """success_rate is 0.5 when half the results succeeded."""
        results = [
            self._make_result("credit_card", "credit_card", success=True),
            self._make_result("credit_card", None, success=False),
        ]
        m = compute_agent_metrics(results)
        assert m["success_rate"] == pytest.approx(0.5)

    def test_success_but_wrong_category(self):
        """category_accuracy is 0.0 when agent succeeds but classifies incorrectly."""
        results = [
            self._make_result("credit_card", "retail_banking", success=True),
        ]
        m = compute_agent_metrics(results)
        assert m["success_rate"] == pytest.approx(1.0)
        assert m["category_accuracy"] == pytest.approx(0.0)

    def test_mixed_success_and_accuracy(self):
        """Accuracy is computed over successful results only."""
        results = [
            self._make_result("credit_card", "credit_card", success=True),  # correct
            self._make_result("retail_banking", "credit_card", success=True),  # wrong
            self._make_result("debt_collection", None, success=False),        # failed
        ]
        m = compute_agent_metrics(results)
        # success_rate = 2/3
        assert m["success_rate"] == pytest.approx(2 / 3)
        # category_accuracy = 1/2 (1 correct out of 2 successful)
        assert m["category_accuracy"] == pytest.approx(0.5)

    def test_output_contains_required_keys(self):
        """Output dict contains success_rate, category_accuracy, success_count, total."""
        results = [self._make_result("credit_card", "credit_card", success=True)]
        m = compute_agent_metrics(results)
        assert "success_rate" in m
        assert "category_accuracy" in m
        assert "success_count" in m
        assert "total" in m

    def test_total_reflects_input_length(self):
        """The 'total' key equals the number of results passed in."""
        results = [self._make_result("credit_card", "credit_card", success=True) for _ in range(7)]
        m = compute_agent_metrics(results)
        assert m["total"] == 7
