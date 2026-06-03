"""Bank customer complaints dataset loader.

Loads the Kaggle ``adhamelkomy/bank-customer-complaint-analysis`` dataset (CC0)
used by the complaint-resolution agent and its evaluation. The relevant file is
``complaints.csv`` with two columns of interest:

- ``product``   — the complaint category label (one of five classes), used as the
  classification ground truth and to derive the gold policy document.
- ``narrative`` — the (pre-processed, lemmatized) complaint text fed to the agent.

The dataset is downloaded on demand via ``kagglehub`` and cached locally; the raw
CSV is never committed to the repository (mirroring the ``knowledge_qa`` loader).
"""

import logging
from pathlib import Path
from typing import cast

import kagglehub
import pandas as pd
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

_KAGGLE_DATASET = "adhamelkomy/bank-customer-complaint-analysis"
_CSV_FILENAME = "complaints.csv"


class BankComplaintExample(BaseModel):
    """A single bank customer complaint example."""

    example_id: int = Field(description="Unique identifier for the example (row index).")
    narrative: str = Field(description="The complaint narrative text (pre-processed/lemmatized).")
    category: str = Field(description="The complaint product category (classification ground truth).")


class ComplaintResolutionOutput(BaseModel):
    """Structured output produced by the complaint-resolution agent.

    The clean, typed fields let deterministic graders read the classification and
    retrieval predictions directly, while the LLM judge scores ``resolution``.
    """

    predicted_category: str = Field(
        description="Predicted complaint category. Must be one of the canonical category labels."
    )
    cited_policy_ids: str = Field(
        default="",
        description="Comma-separated policy document ids the resolution is grounded in (e.g. 'POL-CREDIT-CARD').",
    )
    resolution: str = Field(description="The customer-facing recommended resolution text.")
    reasoning: str = Field(default="", description="Brief justification for the classification and resolution.")


class BankComplaintsDataset:
    """Loader and manager for the bank customer complaints dataset.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory to cache the dataset. If not provided, uses the kagglehub default.

    Examples
    --------
    >>> dataset = BankComplaintsDataset()
    >>> print(f"Total examples: {len(dataset)}")
    >>> example = dataset[0]
    >>> print(example.category, example.narrative[:50])
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._df: pd.DataFrame | None = None
        self._examples: list[BankComplaintExample] | None = None

    def _download_dataset(self) -> Path:
        """Download the dataset using kagglehub and return its directory path."""
        logger.info("Downloading bank customer complaints dataset...")
        path = kagglehub.dataset_download(_KAGGLE_DATASET)
        return Path(path)

    def _load_data(self) -> None:
        """Load and clean the dataset into memory (idempotent)."""
        if self._df is not None:
            return

        dataset_path = self._download_dataset()
        csv_path = dataset_path / _CSV_FILENAME
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Keep only the columns we care about; drop the unnamed index column.
        missing = {"product", "narrative"} - set(df.columns)
        if missing:
            raise ValueError(f"Expected columns missing from {csv_path.name}: {sorted(missing)}")
        df = df[["product", "narrative"]].copy()

        # Drop rows with missing text or label.
        original_count = len(df)
        df = df.dropna(subset=["product", "narrative"])
        df = df[df["narrative"].astype(str).str.strip() != ""]
        df = df.reset_index(drop=True)
        dropped = original_count - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with missing/empty narrative or product")

        self._df = df
        self._examples = [
            BankComplaintExample(example_id=i, narrative=str(narrative), category=str(product))
            for i, (product, narrative) in enumerate(zip(df["product"], df["narrative"], strict=True))
        ]
        logger.info(f"Loaded {len(self._examples)} complaint examples")

    @property
    def dataframe(self) -> pd.DataFrame:
        """Get the cleaned pandas DataFrame (columns: ``product``, ``narrative``)."""
        self._load_data()
        assert self._df is not None
        return self._df

    @property
    def examples(self) -> list[BankComplaintExample]:
        """Get all examples as :class:`BankComplaintExample` objects."""
        self._load_data()
        assert self._examples is not None
        return self._examples

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        self._load_data()
        assert self._examples is not None
        return len(self._examples)

    def __getitem__(self, index: int) -> BankComplaintExample:
        """Get an example by index."""
        self._load_data()
        assert self._examples is not None
        return self._examples[index]

    def get_by_category(self, category: str) -> list[BankComplaintExample]:
        """Get all examples in a specific category."""
        return [ex for ex in self.examples if ex.category == category]

    def get_by_id(self, example_id: int) -> BankComplaintExample | None:
        """Get a single example by its id, or ``None`` if not found."""
        for ex in self.examples:
            if ex.example_id == example_id:
                return ex
        return None

    def get_categories(self) -> list[str]:
        """Get all unique complaint categories present in the dataset."""
        return list(self.dataframe["product"].unique())

    def sample(self, n: int = 10, random_state: int | None = None) -> list[BankComplaintExample]:
        """Get a random sample of ``n`` examples (not class-balanced)."""
        sampled = self.dataframe.sample(n=min(n, len(self)), random_state=random_state)
        return [
            BankComplaintExample(
                example_id=cast(int, row.Index), narrative=str(row.narrative), category=str(row.product)
            )
            for row in sampled.itertuples(index=True)
        ]

    def sample_balanced(self, n_per_category: int = 10, random_state: int | None = None) -> list[BankComplaintExample]:
        """Return a class-balanced sample, up to ``n_per_category`` per category.

        The raw dataset is heavily skewed toward ``credit_reporting`` (~56%), so a
        balanced sample gives every category fair representation in the evaluation set.

        Parameters
        ----------
        n_per_category : int, optional, default=10
            Maximum number of examples to draw from each category.
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        list[BankComplaintExample]
            Sampled examples, grouped by category.
        """
        df = self.dataframe
        frames = [
            group.sample(n=min(n_per_category, len(group)), random_state=random_state)
            for _, group in df.groupby("product")
        ]
        balanced = pd.concat(frames)
        return [
            BankComplaintExample(
                example_id=cast(int, row.Index), narrative=str(row.narrative), category=str(row.product)
            )
            for row in balanced.itertuples(index=True)
        ]
