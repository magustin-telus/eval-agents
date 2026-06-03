"""Shared helpers for complaint-resolution graders."""

from collections.abc import Mapping
from enum import Enum
from typing import Any

from aieng.agent_evals.evaluation import ExperimentItemResult

from ..kb import CATEGORY_LABELS


def get_field(payload: Any, key: str) -> Any:
    """Read ``key`` from dict-like or object payloads."""
    if isinstance(payload, Mapping):
        return payload.get(key)
    return getattr(payload, key, None)


def extract_expected_output(item_result: ExperimentItemResult) -> Any:
    """Extract ``expected_output`` from local-dict or dataset-item structures."""
    item = item_result.item
    if isinstance(item, Mapping):
        return item.get("expected_output")
    return getattr(item, "expected_output", None)


def normalize_category(value: Any) -> str | None:
    """Normalize a category label to its canonical lower-case form.

    Returns ``None`` for empty/missing values. Non-empty values that are not in
    :data:`CATEGORY_LABELS` are returned as-is (lower-cased), so callers can bucket
    them as ``"INVALID"`` for confusion-matrix stability.
    """
    if isinstance(value, Enum):
        value = value.value
    if value is None:
        return None
    token = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    return token or None


def normalize_doc_ids(value: Any) -> set[str]:
    """Normalize policy document ids into a comparable upper-case token set."""
    if value is None:
        return set()

    if isinstance(value, str):
        return {token.strip().upper() for token in value.split(",") if token.strip()}

    if isinstance(value, list | tuple | set):
        normalized: set[str] = set()
        for item in value:
            if item is None:
                continue
            token = str(item).strip().upper()
            if token:
                normalized.add(token)
        return normalized

    token = str(value).strip().upper()
    return {token} if token else set()


__all__ = [
    "CATEGORY_LABELS",
    "extract_expected_output",
    "get_field",
    "normalize_category",
    "normalize_doc_ids",
]
