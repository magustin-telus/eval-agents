"""Tests for complaint_resolution graders._common helpers."""

from types import SimpleNamespace

import pytest
from aieng.agent_evals.complaint_resolution.graders._common import (
    extract_expected_output,
    get_field,
    normalize_category,
    normalize_doc_ids,
)


# ---------------------------------------------------------------------------
# get_field
# ---------------------------------------------------------------------------


class TestGetField:
    """Tests for get_field — reads a key from dict-like or object payloads."""

    def test_reads_key_from_dict(self):
        """Reads a key from a plain dict."""
        assert get_field({"category": "credit_card"}, "category") == "credit_card"

    def test_reads_key_from_mapping(self):
        """Reads a key from any Mapping (not just dict)."""
        from collections import OrderedDict

        m = OrderedDict([("gold_policy_id", "POL-CREDIT-CARD")])
        assert get_field(m, "gold_policy_id") == "POL-CREDIT-CARD"

    def test_returns_none_for_missing_dict_key(self):
        """Returns None when the key is absent from a dict."""
        assert get_field({"a": 1}, "missing") is None

    def test_reads_attr_from_object(self):
        """Reads an attribute from a plain namespace object."""
        obj = SimpleNamespace(predicted_category="debt_collection")
        assert get_field(obj, "predicted_category") == "debt_collection"

    def test_returns_none_for_missing_attr(self):
        """Returns None when the attribute is absent from an object."""
        assert get_field(SimpleNamespace(), "nonexistent") is None

    def test_handles_none_payload(self):
        """Returns None gracefully when the payload itself is None."""
        assert get_field(None, "key") is None

    def test_handles_none_value_in_dict(self):
        """Returns None when the dict key exists but maps to None."""
        assert get_field({"key": None}, "key") is None

    def test_dict_takes_priority_over_attr_path(self):
        """Dict lookup is used when payload is a Mapping, even if it also has attrs."""
        payload = {"predicted_category": "retail_banking"}
        assert get_field(payload, "predicted_category") == "retail_banking"


# ---------------------------------------------------------------------------
# normalize_category
# ---------------------------------------------------------------------------


class TestNormalizeCategory:
    """Tests for normalize_category — canonicalises category label strings."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("credit_card", "credit_card"),
            ("CREDIT_CARD", "credit_card"),
            ("Credit Card", "credit_card"),
            ("credit-card", "credit_card"),
            ("  retail_banking  ", "retail_banking"),
            ("Debt Collection", "debt_collection"),
            ("MORTGAGES_AND_LOANS", "mortgages_and_loans"),
        ],
    )
    def test_normalizes_known_labels(self, value: str, expected: str) -> None:
        """Known category strings are lower-cased, spaces/hyphens replaced with underscores."""
        assert normalize_category(value) == expected

    def test_returns_none_for_none(self):
        """Returns None when input is None."""
        assert normalize_category(None) is None

    def test_returns_none_for_empty_string(self):
        """Returns None for an empty string."""
        assert normalize_category("") is None

    def test_returns_none_for_whitespace_only(self):
        """Returns None for whitespace-only input."""
        assert normalize_category("   ") is None

    def test_unknown_value_returned_lowercased(self):
        """An unrecognised value is returned lower-cased rather than raising."""
        result = normalize_category("something_exotic")
        assert result == "something_exotic"

    def test_enum_value_is_unwrapped(self):
        """Enum values are unwrapped to their .value string before normalisation."""
        from enum import Enum

        class Cat(Enum):
            CC = "Credit Card"

        assert normalize_category(Cat.CC) == "credit_card"

    def test_non_string_coerced_to_string(self):
        """Non-string, non-None inputs are coerced via str()."""
        result = normalize_category(42)  # type: ignore[arg-type]
        assert result == "42"


# ---------------------------------------------------------------------------
# normalize_doc_ids
# ---------------------------------------------------------------------------


class TestNormalizeDocIds:
    """Tests for normalize_doc_ids — returns a set of upper-case policy id tokens."""

    def test_returns_empty_set_for_none(self):
        """Returns an empty set when input is None."""
        assert normalize_doc_ids(None) == set()

    def test_returns_empty_set_for_empty_string(self):
        """Returns an empty set for an empty string."""
        assert normalize_doc_ids("") == set()

    def test_returns_empty_set_for_whitespace_string(self):
        """Returns an empty set for a whitespace-only string."""
        assert normalize_doc_ids("   ") == set()

    def test_single_id_from_string(self):
        """Parses a single policy id from a plain string."""
        assert normalize_doc_ids("POL-CREDIT-CARD") == {"POL-CREDIT-CARD"}

    def test_lowercased_id_is_uppercased(self):
        """Lower-case ids are upper-cased."""
        assert normalize_doc_ids("pol-credit-card") == {"POL-CREDIT-CARD"}

    def test_comma_separated_string(self):
        """Comma-separated string is split into multiple ids."""
        result = normalize_doc_ids("POL-CREDIT-CARD, POL-RETAIL-BANKING")
        assert result == {"POL-CREDIT-CARD", "POL-RETAIL-BANKING"}

    def test_extra_whitespace_stripped(self):
        """Leading/trailing whitespace around each id is stripped."""
        result = normalize_doc_ids("  POL-CREDIT-CARD , POL-DEBT-COLLECTION  ")
        assert result == {"POL-CREDIT-CARD", "POL-DEBT-COLLECTION"}

    def test_list_of_ids(self):
        """A list of id strings is converted to a set."""
        result = normalize_doc_ids(["POL-CREDIT-CARD", "pol-retail-banking"])
        assert result == {"POL-CREDIT-CARD", "POL-RETAIL-BANKING"}

    def test_list_with_none_entries(self):
        """None entries inside a list are silently ignored."""
        result = normalize_doc_ids(["POL-CREDIT-CARD", None, "POL-RETAIL-BANKING"])
        assert result == {"POL-CREDIT-CARD", "POL-RETAIL-BANKING"}

    def test_set_of_ids(self):
        """A set of id strings is normalised correctly."""
        result = normalize_doc_ids({"pol-credit-card", "pol-debt-collection"})
        assert result == {"POL-CREDIT-CARD", "POL-DEBT-COLLECTION"}

    def test_tuple_of_ids(self):
        """A tuple of id strings is normalised correctly."""
        result = normalize_doc_ids(("POL-CREDIT-CARD",))
        assert result == {"POL-CREDIT-CARD"}

    def test_non_string_scalar_coerced(self):
        """A non-string, non-iterable scalar is coerced via str() to a singleton set."""
        result = normalize_doc_ids(123)  # type: ignore[arg-type]
        assert result == {"123"}

    def test_empty_list_returns_empty_set(self):
        """An empty list returns an empty set."""
        assert normalize_doc_ids([]) == set()

    def test_comma_separated_with_empty_tokens_ignored(self):
        """Empty tokens produced by splitting (e.g. trailing comma) are ignored."""
        result = normalize_doc_ids("POL-CREDIT-CARD,")
        assert result == {"POL-CREDIT-CARD"}


# ---------------------------------------------------------------------------
# extract_expected_output
# ---------------------------------------------------------------------------


class TestExtractExpectedOutput:
    """Tests for extract_expected_output — pulls expected_output from item results."""

    def _make_item_result(self, item: object) -> object:
        """Wrap an item in a minimal ExperimentItemResult-like namespace."""
        return SimpleNamespace(item=item)

    def test_reads_from_dict_item(self):
        """Reads expected_output from a dict-style item."""
        item = {"expected_output": {"category": "credit_card"}, "input": "narrative"}
        result = extract_expected_output(self._make_item_result(item))
        assert result == {"category": "credit_card"}

    def test_reads_from_object_item(self):
        """Reads expected_output from an object-style item."""
        item = SimpleNamespace(expected_output={"category": "debt_collection"})
        result = extract_expected_output(self._make_item_result(item))
        assert result == {"category": "debt_collection"}

    def test_returns_none_when_key_absent_from_dict(self):
        """Returns None when expected_output is absent from a dict item."""
        result = extract_expected_output(self._make_item_result({"input": "x"}))
        assert result is None

    def test_returns_none_when_attr_absent_from_object(self):
        """Returns None when expected_output attr is absent from an object item."""
        result = extract_expected_output(self._make_item_result(SimpleNamespace()))
        assert result is None

    def test_returns_none_value_from_dict(self):
        """Returns None when the dict key maps to None."""
        result = extract_expected_output(self._make_item_result({"expected_output": None}))
        assert result is None
