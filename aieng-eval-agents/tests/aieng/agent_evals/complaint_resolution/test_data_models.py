"""Tests for complaint_resolution data models and dataset utilities.

Covers:
- ComplaintResolutionOutput parsing and validation
- BankComplaintExample construction
- BankComplaintsDataset.sample_balanced class-balance guarantee (without network I/O)
- PolicyKnowledgeBase._parse_policy_markdown (pure file parsing)
- PolicyKnowledgeBase._content_hash determinism and sensitivity
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from aieng.agent_evals.complaint_resolution.data.bank_complaints import (
    BankComplaintExample,
    BankComplaintsDataset,
    ComplaintResolutionOutput,
)
from aieng.agent_evals.complaint_resolution.kb import (
    CATEGORY_LABELS,
    CATEGORY_TO_POLICY_ID,
    PolicyKnowledgeBase,
    _parse_policy_markdown,
)


# ---------------------------------------------------------------------------
# ComplaintResolutionOutput
# ---------------------------------------------------------------------------


class TestComplaintResolutionOutput:
    """Tests for the ComplaintResolutionOutput Pydantic model."""

    def test_round_trips_via_model_dump(self):
        """model_dump() round-trips through model_validate()."""
        output = ComplaintResolutionOutput(
            predicted_category="credit_card",
            cited_policy_ids="POL-CREDIT-CARD",
            resolution="We will investigate your dispute.",
            reasoning="The complaint mentions a credit card charge.",
        )
        dumped = output.model_dump()
        restored = ComplaintResolutionOutput.model_validate(dumped)
        assert restored.predicted_category == "credit_card"
        assert restored.cited_policy_ids == "POL-CREDIT-CARD"
        assert restored.resolution == "We will investigate your dispute."
        assert restored.reasoning == "The complaint mentions a credit card charge."

    def test_round_trips_via_json(self):
        """model_validate_json(model_dump_json()) round-trips correctly."""
        output = ComplaintResolutionOutput(
            predicted_category="retail_banking",
            cited_policy_ids="POL-RETAIL-BANKING",
            resolution="Your account issue has been escalated.",
            reasoning="Retail banking complaint.",
        )
        json_str = output.model_dump_json()
        restored = ComplaintResolutionOutput.model_validate_json(json_str)
        assert restored.predicted_category == "retail_banking"

    def test_optional_fields_default_to_empty_string(self):
        """cited_policy_ids and reasoning default to empty string when omitted."""
        output = ComplaintResolutionOutput(
            predicted_category="debt_collection",
            resolution="We will contact the collector on your behalf.",
        )
        assert output.cited_policy_ids == ""
        assert output.reasoning == ""

    def test_all_five_categories_are_valid_values(self):
        """ComplaintResolutionOutput accepts each of the five canonical categories."""
        for cat in CATEGORY_LABELS:
            obj = ComplaintResolutionOutput(
                predicted_category=cat,
                resolution="resolution text",
            )
            assert obj.predicted_category == cat

    def test_missing_required_field_raises(self):
        """Omitting a required field raises a ValidationError."""
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            ComplaintResolutionOutput.model_validate({"resolution": "text"})  # missing predicted_category

    def test_model_dump_contains_all_fields(self):
        """model_dump() contains exactly the four expected keys."""
        output = ComplaintResolutionOutput(
            predicted_category="credit_reporting",
            resolution="We'll dispute the entry.",
        )
        keys = set(output.model_dump().keys())
        assert keys == {"predicted_category", "cited_policy_ids", "resolution", "reasoning"}


# ---------------------------------------------------------------------------
# BankComplaintExample
# ---------------------------------------------------------------------------


class TestBankComplaintExample:
    """Tests for the BankComplaintExample Pydantic model."""

    def test_construction(self):
        """BankComplaintExample stores all three fields correctly."""
        ex = BankComplaintExample(example_id=0, narrative="charged twice credit card", category="credit_card")
        assert ex.example_id == 0
        assert ex.narrative == "charged twice credit card"
        assert ex.category == "credit_card"

    def test_model_validates_from_dict(self):
        """model_validate() accepts a plain dict."""
        ex = BankComplaintExample.model_validate(
            {"example_id": 1, "narrative": "debt collector called", "category": "debt_collection"}
        )
        assert ex.category == "debt_collection"


# ---------------------------------------------------------------------------
# BankComplaintsDataset — class-balance (monkeypatched, no network)
# ---------------------------------------------------------------------------


def _make_fake_df():
    """Return a minimal fake DataFrame with five balanced categories."""
    import pandas as pd

    rows = []
    for cat in CATEGORY_LABELS:
        for i in range(4):
            rows.append({"product": cat, "narrative": f"{cat} narrative {i}"})
    return pd.DataFrame(rows)


class TestBankComplaintsDatasetBalanced:
    """Tests for BankComplaintsDataset.sample_balanced (no Kaggle network call)."""

    @pytest.fixture()
    def dataset(self) -> BankComplaintsDataset:
        """Return a BankComplaintsDataset pre-loaded with a fake DataFrame."""
        import pandas as pd

        ds = BankComplaintsDataset.__new__(BankComplaintsDataset)
        fake_df = _make_fake_df()
        ds._cache_dir = None  # type: ignore[attr-defined]
        ds._df = fake_df
        ds._examples = [
            BankComplaintExample(
                example_id=i,
                narrative=str(row["narrative"]),
                category=str(row["product"]),
            )
            for i, row in fake_df.iterrows()
        ]
        return ds

    def test_sample_balanced_returns_correct_total(self, dataset: BankComplaintsDataset):
        """sample_balanced returns n_per_category × n_categories examples."""
        n = 2
        examples = dataset.sample_balanced(n_per_category=n)
        assert len(examples) == n * len(CATEGORY_LABELS)

    def test_sample_balanced_all_categories_present(self, dataset: BankComplaintsDataset):
        """sample_balanced returns at least one example per category."""
        examples = dataset.sample_balanced(n_per_category=2)
        cats = {e.category for e in examples}
        assert cats == set(CATEGORY_LABELS)

    def test_sample_balanced_no_category_over_represented(self, dataset: BankComplaintsDataset):
        """No category has more than n_per_category examples."""
        from collections import Counter

        n = 2
        examples = dataset.sample_balanced(n_per_category=n)
        counts = Counter(e.category for e in examples)
        assert max(counts.values()) <= n

    def test_sample_balanced_reproducible_with_same_seed(self, dataset: BankComplaintsDataset):
        """same random_state produces the same sample."""
        a = dataset.sample_balanced(n_per_category=2, random_state=0)
        b = dataset.sample_balanced(n_per_category=2, random_state=0)
        assert [e.example_id for e in a] == [e.example_id for e in b]

    def test_sample_balanced_different_seeds_may_differ(self, dataset: BankComplaintsDataset):
        """Different random_state values are not guaranteed to produce the same order."""
        a = dataset.sample_balanced(n_per_category=2, random_state=0)
        b = dataset.sample_balanced(n_per_category=2, random_state=99)
        # They might differ; we just check both have the right length.
        assert len(a) == len(b) == 2 * len(CATEGORY_LABELS)

    def test_sample_balanced_capped_at_available(self, dataset: BankComplaintsDataset):
        """Requesting more than available per category returns what's there."""
        examples = dataset.sample_balanced(n_per_category=1000)
        from collections import Counter

        counts = Counter(e.category for e in examples)
        # Each category only has 4 rows in the fixture
        assert all(v <= 4 for v in counts.values())

    def test_get_by_category_filters_correctly(self, dataset: BankComplaintsDataset):
        """get_by_category returns only examples with the requested category."""
        results = dataset.get_by_category("credit_card")
        assert all(e.category == "credit_card" for e in results)
        assert len(results) == 4  # 4 rows per category in fixture

    def test_len_returns_total_example_count(self, dataset: BankComplaintsDataset):
        """len(dataset) returns the total number of examples."""
        assert len(dataset) == len(CATEGORY_LABELS) * 4

    def test_getitem_returns_correct_example(self, dataset: BankComplaintsDataset):
        """dataset[0] returns the first BankComplaintExample."""
        ex = dataset[0]
        assert isinstance(ex, BankComplaintExample)

    def test_get_categories_returns_all_categories(self, dataset: BankComplaintsDataset):
        """get_categories() returns all five canonical category labels."""
        cats = set(dataset.get_categories())
        assert cats == set(CATEGORY_LABELS)


# ---------------------------------------------------------------------------
# CATEGORY_LABELS / CATEGORY_TO_POLICY_ID constants
# ---------------------------------------------------------------------------


class TestCategoryConstants:
    """Tests for the CATEGORY_LABELS and CATEGORY_TO_POLICY_ID module constants."""

    def test_category_labels_is_tuple_of_five(self):
        """CATEGORY_LABELS is a tuple of exactly five strings."""
        assert isinstance(CATEGORY_LABELS, tuple)
        assert len(CATEGORY_LABELS) == 5

    def test_category_to_policy_id_covers_all_labels(self):
        """Every category in CATEGORY_LABELS has an entry in CATEGORY_TO_POLICY_ID."""
        for cat in CATEGORY_LABELS:
            assert cat in CATEGORY_TO_POLICY_ID, f"{cat!r} missing from CATEGORY_TO_POLICY_ID"

    def test_policy_ids_are_uppercase_with_pol_prefix(self):
        """Every policy id starts with 'POL-' and is upper-cased."""
        for cat, pol_id in CATEGORY_TO_POLICY_ID.items():
            assert pol_id.startswith("POL-"), f"{cat}: policy id {pol_id!r} does not start with 'POL-'"
            assert pol_id == pol_id.upper(), f"{cat}: policy id {pol_id!r} is not upper-cased"

    def test_policy_ids_are_unique(self):
        """Each category maps to a distinct policy id."""
        ids = list(CATEGORY_TO_POLICY_ID.values())
        assert len(ids) == len(set(ids)), "Duplicate policy ids in CATEGORY_TO_POLICY_ID"


# ---------------------------------------------------------------------------
# _parse_policy_markdown (pure function — no KB or embedding calls)
# ---------------------------------------------------------------------------


class TestParsePolicyMarkdown:
    """Tests for the _parse_policy_markdown helper in kb.py."""

    def _write_policy(self, tmp_path: Path, contents: str) -> Path:
        p = tmp_path / "policy.md"
        p.write_text(contents, encoding="utf-8")
        return p

    def test_parses_all_three_front_matter_fields(self, tmp_path: Path):
        """id, title, and category are parsed from standard front-matter lines."""
        contents = (
            "# id: POL-CREDIT-CARD\n"
            "# title: Credit Card Complaint Resolution Policy\n"
            "# category: credit_card\n"
            "\n"
            "This policy governs credit card disputes.\n"
        )
        doc = _parse_policy_markdown(self._write_policy(tmp_path, contents))
        assert doc.id == "POL-CREDIT-CARD"
        assert doc.title == "Credit Card Complaint Resolution Policy"
        assert doc.category == "credit_card"

    def test_parses_body_text(self, tmp_path: Path):
        """Body text after front-matter is captured in doc.text."""
        contents = (
            "# id: POL-RETAIL-BANKING\n"
            "# title: Retail Banking Policy\n"
            "# category: retail_banking\n"
            "\n"
            "Section 1: Eligibility\n"
            "Customers must have a valid account.\n"
        )
        doc = _parse_policy_markdown(self._write_policy(tmp_path, contents))
        assert "Section 1: Eligibility" in doc.text
        assert "Customers must have a valid account." in doc.text

    def test_score_is_none_by_default(self, tmp_path: Path):
        """Parsed doc has score=None (no retrieval context)."""
        contents = (
            "# id: POL-DEBT-COLLECTION\n"
            "# title: Debt Collection Policy\n"
            "# category: debt_collection\n"
            "\n"
            "Body.\n"
        )
        doc = _parse_policy_markdown(self._write_policy(tmp_path, contents))
        assert doc.score is None

    def test_missing_front_matter_field_raises_value_error(self, tmp_path: Path):
        """Raises ValueError when any required front-matter field is missing."""
        contents = (
            "# id: POL-CREDIT-CARD\n"
            "# title: Credit Card Policy\n"
            "# (no category line)\n"
            "\n"
            "Body.\n"
        )
        with pytest.raises(ValueError, match="category"):
            _parse_policy_markdown(self._write_policy(tmp_path, contents))

    def test_case_insensitive_front_matter_keys(self, tmp_path: Path):
        """Front-matter keys are matched case-insensitively (e.g. ID:, Title:)."""
        contents = (
            "# ID: POL-MORTGAGES\n"
            "# Title: Mortgages Policy\n"
            "# Category: mortgages_and_loans\n"
            "\n"
            "Body text.\n"
        )
        doc = _parse_policy_markdown(self._write_policy(tmp_path, contents))
        assert doc.id == "POL-MORTGAGES"
        assert doc.category == "mortgages_and_loans"

    def test_blank_lines_before_body_are_ignored(self, tmp_path: Path):
        """Multiple blank lines between front-matter and body are allowed."""
        contents = (
            "# id: POL-CREDIT-REPORTING\n"
            "# title: Credit Reporting Policy\n"
            "# category: credit_reporting\n"
            "\n\n\n"
            "Policy body starts here.\n"
        )
        doc = _parse_policy_markdown(self._write_policy(tmp_path, contents))
        assert "Policy body starts here." in doc.text

    def test_actual_policy_files_parse_without_error(self):
        """All five shipped policy markdown files parse without raising."""
        policies_dir = (
            Path(__file__).parents[4]
            / "aieng"
            / "agent_evals"
            / "complaint_resolution"
            / "data"
            / "policies"
        )
        policy_files = list(policies_dir.glob("*.md"))
        assert len(policy_files) >= 5, "Expected at least 5 policy files"
        for path in policy_files:
            doc = _parse_policy_markdown(path)
            assert doc.id
            assert doc.title
            assert doc.category
            assert doc.text


# ---------------------------------------------------------------------------
# PolicyKnowledgeBase — _content_hash (no embedding network call)
# ---------------------------------------------------------------------------


class TestKnowledgeBaseContentHash:
    """Tests for PolicyKnowledgeBase._content_hash (pure, no I/O beyond file reads)."""

    def _make_kb(self, policies_dir: Path) -> PolicyKnowledgeBase:
        """Construct a KB over a custom policies dir without making embedding calls."""
        mock_config = MagicMock()
        mock_config.embedding_model_name = "test-embedding-model"
        with patch(
            "aieng.agent_evals.complaint_resolution.kb.EmbeddingClient",
            return_value=MagicMock(model="test-embedding-model"),
        ):
            return PolicyKnowledgeBase(config=mock_config, policies_dir=policies_dir)

    def _write_policy(self, path: Path, pol_id: str, category: str, body: str) -> None:
        path.write_text(
            f"# id: {pol_id}\n# title: {pol_id} Title\n# category: {category}\n\n{body}\n",
            encoding="utf-8",
        )

    def test_hash_is_deterministic(self, tmp_path: Path):
        """Same KB produces the same hash on repeated calls."""
        self._write_policy(tmp_path / "a.md", "POL-A", "credit_card", "Body A.")
        kb = self._make_kb(tmp_path)
        assert kb._content_hash() == kb._content_hash()

    def test_hash_changes_when_body_changes(self, tmp_path: Path):
        """Hash changes when a policy document's body is modified."""
        p = tmp_path / "a.md"
        self._write_policy(p, "POL-A", "credit_card", "Original body.")
        kb1 = self._make_kb(tmp_path)
        h1 = kb1._content_hash()

        self._write_policy(p, "POL-A", "credit_card", "Changed body.")
        kb2 = self._make_kb(tmp_path)
        h2 = kb2._content_hash()

        assert h1 != h2

    def test_hash_changes_when_doc_added(self, tmp_path: Path):
        """Hash changes when a new policy document is added to the directory."""
        self._write_policy(tmp_path / "a.md", "POL-A", "credit_card", "Body A.")
        kb1 = self._make_kb(tmp_path)
        h1 = kb1._content_hash()

        self._write_policy(tmp_path / "b.md", "POL-B", "retail_banking", "Body B.")
        kb2 = self._make_kb(tmp_path)
        h2 = kb2._content_hash()

        assert h1 != h2

    def test_hash_is_16_hex_chars(self, tmp_path: Path):
        """Hash is a 16-character hex string (first 16 digits of SHA-256)."""
        self._write_policy(tmp_path / "a.md", "POL-A", "credit_card", "Body.")
        kb = self._make_kb(tmp_path)
        h = kb._content_hash()
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_docs_property_returns_copy(self, tmp_path: Path):
        """kb.docs returns a copy; mutating it does not affect the internal list."""
        self._write_policy(tmp_path / "a.md", "POL-A", "credit_card", "Body.")
        kb = self._make_kb(tmp_path)
        docs_copy = kb.docs
        docs_copy.clear()
        assert len(kb.docs) > 0

    def test_get_doc_returns_matching_doc(self, tmp_path: Path):
        """get_doc('POL-A') returns the PolicyDoc with id 'POL-A'."""
        self._write_policy(tmp_path / "a.md", "POL-A", "credit_card", "Body.")
        kb = self._make_kb(tmp_path)
        doc = kb.get_doc("POL-A")
        assert doc is not None
        assert doc.id == "POL-A"

    def test_get_doc_returns_none_for_unknown_id(self, tmp_path: Path):
        """get_doc returns None when the id does not exist."""
        self._write_policy(tmp_path / "a.md", "POL-A", "credit_card", "Body.")
        kb = self._make_kb(tmp_path)
        assert kb.get_doc("DOES-NOT-EXIST") is None

    def test_empty_policies_dir_raises(self, tmp_path: Path):
        """KB raises FileNotFoundError when the policies directory has no .md files."""
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            self._make_kb(empty)
