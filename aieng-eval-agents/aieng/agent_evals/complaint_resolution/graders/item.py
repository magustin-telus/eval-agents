"""Item-level deterministic graders for complaint-resolution outputs.

Scores one complaint prediction against its ground truth using pure rules (no LLM):
classification correctness and policy-retrieval correctness. These complement the
LLM-judge (resolution quality) and trace (groundedness/efficiency) evaluators.
"""

from typing import Any

from aieng.agent_evals.evaluation import Evaluation

from ._common import get_field, normalize_category, normalize_doc_ids


def item_level_deterministic_grader(
    input: Any,  # noqa: A002
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """Evaluate one complaint prediction using deterministic rules.

    Parameters
    ----------
    input : Any
        Item input payload (complaint narrative). Unused; part of the interface.
    output : Any
        Model output payload, expected to contain ``predicted_category`` and
        ``cited_policy_ids``.
    expected_output : Any
        Ground-truth payload, expected to contain ``category`` and ``gold_policy_id``.
    metadata : dict[str, Any] | None, optional
        Item metadata from the dataset. Unused by this grader.
    **kwargs : Any
        Additional evaluator kwargs. Ignored.

    Returns
    -------
    list[Evaluation]
        Deterministic per-item metrics:
        ``category_correct``, ``policy_match``, ``retrieval_precision``,
        ``retrieval_coverage``.
    """
    del input, metadata, kwargs  # Unused but part of the evaluator interface.

    # --- Classification correctness ---
    expected_category = normalize_category(get_field(expected_output, "category"))
    predicted_category = normalize_category(get_field(output, "predicted_category"))
    category_correct = expected_category is not None and predicted_category == expected_category

    # --- Retrieval correctness ---
    gold_ids = normalize_doc_ids(get_field(expected_output, "gold_policy_id"))
    predicted_ids = normalize_doc_ids(get_field(output, "cited_policy_ids"))

    true_positive_ids = gold_ids & predicted_ids
    tp_count = len(true_positive_ids)
    predicted_count = len(predicted_ids)
    gold_count = len(gold_ids)

    # Did the agent cite the correct (gold) policy at all?
    policy_match = bool(gold_ids) and gold_ids.issubset(predicted_ids)

    # Precision: of the cited policies, how many were gold? Coverage: of the gold
    # policies, how many were cited?
    retrieval_precision = float(tp_count) / float(predicted_count) if predicted_count else 0.0
    retrieval_coverage = float(tp_count) / float(gold_count) if gold_count else 0.0

    return [
        Evaluation(
            name="category_correct",
            value=1.0 if category_correct else 0.0,
            metadata={"expected": expected_category, "actual": predicted_category},
        ),
        Evaluation(
            name="policy_match",
            value=1.0 if policy_match else 0.0,
            metadata={"gold": sorted(gold_ids), "predicted": sorted(predicted_ids)},
        ),
        Evaluation(
            name="retrieval_precision",
            value=retrieval_precision,
            metadata={"true_positive_count": tp_count, "predicted_count": predicted_count},
        ),
        Evaluation(
            name="retrieval_coverage",
            value=retrieval_coverage,
            metadata={"true_positive_count": tp_count, "gold_count": gold_count},
        ),
    ]


__all__ = ["item_level_deterministic_grader"]
