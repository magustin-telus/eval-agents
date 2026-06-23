"""Reusable functions for dataset analysis and exploration.

Provides utilities for:
- Dataset statistics and distribution
- Retrieval baseline evaluation
- Agent smoke testing on samples
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .data import BankComplaintsDataset
from .kb import CATEGORY_TO_POLICY_ID, PolicyKnowledgeBase
from .task import ComplaintResolutionTask


@dataclass
class CategoryStats:
    """Statistics for a complaint category."""

    name: str
    count: int
    percentage: float
    avg_narrative_length: float
    min_narrative_length: int
    max_narrative_length: int


@dataclass
class RetrievalResult:
    """Result of retrieval test for one example."""

    example_id: int
    category: str
    gold_policy_id: str
    retrieved_ids: list[str]
    gold_rank: int | None  # 1-indexed if found, None if not in top-k


@dataclass
class AgentTestResult:
    """Result of agent test on one example."""

    example_id: int
    gold_category: str
    predicted_category: str | None
    cited_policy_ids: str
    success: bool
    error: str | None = None


def get_dataset_statistics(sample_size: int | None = None) -> tuple[dict[str, CategoryStats], list[int]]:
    """Get dataset statistics by category.

    Parameters
    ----------
    sample_size : int | None
        If provided, analyze only first N narratives; else analyze all.

    Returns
    -------
    tuple[dict[str, CategoryStats], list[int]]
        Dictionary of category stats and list of narrative lengths.
    """
    dataset = BankComplaintsDataset()
    size = min(sample_size, len(dataset)) if sample_size else len(dataset)

    categories = []
    narrative_lengths = []

    for i in range(size):
        example = dataset[i]
        categories.append(example.category)
        narrative_lengths.append(len(example.narrative.split()))

    category_counts = Counter(categories)
    total = len(dataset)

    stats: dict[str, CategoryStats] = {}
    for category, count in category_counts.items():
        # Get narrative lengths for this category
        lengths = [len(dataset[i].narrative.split()) for i in range(len(dataset)) if dataset[i].category == category]
        stats[category] = CategoryStats(
            name=category,
            count=count,
            percentage=(count / total) * 100,
            avg_narrative_length=np.mean(lengths) if lengths else 0,
            min_narrative_length=min(lengths) if lengths else 0,
            max_narrative_length=max(lengths) if lengths else 0,
        )

    return stats, narrative_lengths


async def test_retrieval_on_sample(n_per_category: int = 2, random_state: int = 42) -> dict[str, list[RetrievalResult]]:
    """Test retrieval quality on balanced sample.

    Parameters
    ----------
    n_per_category : int
        Number of examples per category to test.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, list[RetrievalResult]]
        Results indexed by category.
    """
    dataset = BankComplaintsDataset()
    kb = PolicyKnowledgeBase()

    balanced = dataset.sample_balanced(n_per_category=n_per_category, random_state=random_state)

    results_by_category: dict[str, list[RetrievalResult]] = {}

    for example in balanced:
        if example.category not in results_by_category:
            results_by_category[example.category] = []

        retrieved = await kb.retrieve(example.narrative, k=3)
        retrieved_ids = [r.id for r in retrieved]
        gold_id = CATEGORY_TO_POLICY_ID.get(example.category)

        rank = None
        if gold_id in retrieved_ids:
            rank = retrieved_ids.index(gold_id) + 1  # 1-indexed

        result = RetrievalResult(
            example_id=example.example_id,
            category=example.category,
            gold_policy_id=gold_id or "UNKNOWN",
            retrieved_ids=retrieved_ids,
            gold_rank=rank,
        )
        results_by_category[example.category].append(result)

    return results_by_category


async def test_agent_on_sample(n_per_category: int = 1, random_state: int = 42) -> list[AgentTestResult]:
    """Run agent on balanced sample.

    Parameters
    ----------
    n_per_category : int
        Number of examples per category to test.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    list[AgentTestResult]
        Test results for each example.
    """
    dataset = BankComplaintsDataset()
    balanced = dataset.sample_balanced(n_per_category=n_per_category, random_state=random_state)

    task = ComplaintResolutionTask()
    results = []

    for example in balanced:
        try:
            result = await task(item={"input": example.narrative})
            if result:
                predicted_cat = result.get("predicted_category")
                cited_policies = result.get("cited_policy_ids", "")
                results.append(
                    AgentTestResult(
                        example_id=example.example_id,
                        gold_category=example.category,
                        predicted_category=predicted_cat,
                        cited_policy_ids=cited_policies,
                        success=True,
                    )
                )
            else:
                results.append(
                    AgentTestResult(
                        example_id=example.example_id,
                        gold_category=example.category,
                        predicted_category=None,
                        cited_policy_ids="",
                        success=False,
                        error="No output",
                    )
                )
        except Exception as e:
            results.append(
                AgentTestResult(
                    example_id=example.example_id,
                    gold_category=example.category,
                    predicted_category=None,
                    cited_policy_ids="",
                    success=False,
                    error=str(e),
                )
            )

    await task.close()
    return results


def compute_retrieval_metrics(results: dict[str, list[RetrievalResult]]) -> dict:
    """Compute retrieval metrics from test results.

    Parameters
    ----------
    results : dict[str, list[RetrievalResult]]
        Results by category.

    Returns
    -------
    dict
        Metrics: precision@3, recall, coverage by category.
    """
    metrics = {}

    for category, category_results in results.items():
        found = sum(1 for r in category_results if r.gold_rank is not None)
        total = len(category_results)
        metrics[category] = {
            "precision_at_3": found / total if total > 0 else 0,
            "found_count": found,
            "total": total,
            "avg_rank": np.mean([r.gold_rank for r in category_results if r.gold_rank]) if found > 0 else None,
        }

    return metrics


def compute_agent_metrics(results: list[AgentTestResult]) -> dict:
    """Compute agent test metrics.

    Parameters
    ----------
    results : list[AgentTestResult]
        Test results.

    Returns
    -------
    dict
        Success rate and category accuracy.
    """
    success_count = sum(1 for r in results if r.success)
    if success_count == 0:
        return {"success_rate": 0, "category_accuracy": 0, "success_count": 0, "total": len(results)}

    correct_count = sum(1 for r in results if r.success and r.gold_category == r.predicted_category)
    return {
        "success_rate": success_count / len(results),
        "category_accuracy": correct_count / success_count if success_count > 0 else 0,
        "success_count": success_count,
        "correct_count": correct_count,
        "total": len(results),
    }
