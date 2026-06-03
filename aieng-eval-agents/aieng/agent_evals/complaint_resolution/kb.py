"""In-process vector store over the complaint-resolution policy knowledge base.

The knowledge base is a small set of hand-authored policy documents (one per
complaint category) stored as markdown under ``data/policies/``. This module
loads those documents, embeds them via
:class:`aieng.agent_evals.complaint_resolution.embeddings.EmbeddingClient`, and
serves nearest-neighbour retrieval using cosine similarity over a NumPy matrix.

The KB is tiny (one doc per category), so an in-process NumPy index is more than
sufficient — no external vector database is required. Embeddings are cached to a
``.npz`` file keyed by a hash of the document contents and the embedding model, so
the embedding endpoint is only hit when the documents or model change.
"""

import hashlib
import logging
import re
import tempfile
from pathlib import Path

import numpy as np
from aieng.agent_evals.configs import Configs
from pydantic import BaseModel, Field

from .embeddings import EmbeddingClient


logger = logging.getLogger(__name__)

# Canonical complaint categories — these match the `product` labels in the
# bank-customer-complaint dataset exactly (no taxonomy collapsing needed).
CATEGORY_LABELS: tuple[str, ...] = (
    "credit_card",
    "credit_reporting",
    "debt_collection",
    "mortgages_and_loans",
    "retail_banking",
)

# Maps each dataset category label to the gold policy document id. This is both
# the classification target set and the gold retrieval target (one doc per category).
CATEGORY_TO_POLICY_ID: dict[str, str] = {
    "credit_card": "POL-CREDIT-CARD",
    "credit_reporting": "POL-CREDIT-REPORTING",
    "debt_collection": "POL-DEBT-COLLECTION",
    "mortgages_and_loans": "POL-MORTGAGE-LOANS",
    "retail_banking": "POL-RETAIL-BANKING",
}

_DEFAULT_POLICIES_DIR = Path(__file__).parent / "data" / "policies"
_FRONT_MATTER_RE = re.compile(r"^#\s*(id|title|category)\s*:\s*(.+?)\s*$", re.IGNORECASE)


class PolicyDoc(BaseModel):
    """A single policy knowledge-base document.

    Parameters
    ----------
    id : str
        Stable policy identifier (e.g. ``"POL-CREDIT-CARD"``).
    title : str
        Human-readable policy title.
    category : str
        Complaint category this policy applies to.
    text : str
        Full policy body text (front matter stripped).
    score : float | None, optional
        Cosine similarity to a query. Populated only on retrieval results.
    """

    id: str
    title: str
    category: str
    text: str
    score: float | None = Field(default=None)


def _parse_policy_markdown(path: Path) -> PolicyDoc:
    """Parse a policy markdown file into a :class:`PolicyDoc`.

    Front matter is expressed as leading ``# id:`` / ``# title:`` / ``# category:``
    comment lines; everything after them is the policy body.

    Parameters
    ----------
    path : Path
        Path to the markdown file.

    Returns
    -------
    PolicyDoc
        The parsed document (without a score).

    Raises
    ------
    ValueError
        If any of the required front-matter fields are missing.
    """
    fields: dict[str, str] = {}
    body_lines: list[str] = []
    in_body = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not in_body:
            match = _FRONT_MATTER_RE.match(line)
            if match:
                fields[match.group(1).lower()] = match.group(2).strip()
                continue
            if line.strip() == "":
                continue
            # First non-front-matter, non-blank line begins the body.
            in_body = True
        body_lines.append(line)

    missing = {"id", "title", "category"} - fields.keys()
    if missing:
        raise ValueError(f"Policy file {path.name} is missing front-matter fields: {sorted(missing)}")

    return PolicyDoc(
        id=fields["id"],
        title=fields["title"],
        category=fields["category"],
        text="\n".join(body_lines).strip(),
    )


class PolicyKnowledgeBase:
    """Load, embed, and retrieve policy documents via cosine similarity.

    Parameters
    ----------
    config : Configs | None, optional
        Configuration object. If ``None``, a default ``Configs()`` is created.
    policies_dir : str | Path | None, optional
        Directory containing the policy markdown files. Defaults to the package's
        ``data/policies/`` directory.
    cache_dir : str | Path | None, optional
        Directory for the embedding cache ``.npz``. Defaults to a subdirectory of
        the system temp directory.

    Examples
    --------
    >>> kb = PolicyKnowledgeBase()
    >>> results = await kb.retrieve("my credit card was charged twice", k=2)
    >>> results[0].id
    'POL-CREDIT-CARD'
    >>> await kb.aclose()
    """

    def __init__(
        self,
        config: Configs | None = None,
        policies_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self._config = config or Configs()  # type: ignore[call-arg]
        self._policies_dir = Path(policies_dir) if policies_dir else _DEFAULT_POLICIES_DIR
        self._cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "complaint_policy_kb"
        self._embedder = EmbeddingClient(config=self._config)
        self._docs: list[PolicyDoc] = self._load_docs()
        self._matrix: np.ndarray | None = None  # (n_docs, dim), L2-normalized rows

    def _load_docs(self) -> list[PolicyDoc]:
        """Load and parse all policy markdown files, sorted by id for determinism."""
        files = sorted(self._policies_dir.glob("*.md"))
        if not files:
            raise FileNotFoundError(f"No policy markdown files found in {self._policies_dir}")
        docs = [_parse_policy_markdown(p) for p in files]
        docs.sort(key=lambda d: d.id)
        logger.info("Loaded %d policy documents from %s", len(docs), self._policies_dir)
        return docs

    @property
    def docs(self) -> list[PolicyDoc]:
        """The loaded policy documents (without scores)."""
        return list(self._docs)

    def get_doc(self, policy_id: str) -> PolicyDoc | None:
        """Return the policy document with the given id, or ``None``."""
        return next((d for d in self._docs if d.id == policy_id), None)

    def _content_hash(self) -> str:
        """Hash document contents + embedding model to key the embedding cache."""
        hasher = hashlib.sha256()
        hasher.update(self._embedder.model.encode("utf-8"))
        for doc in self._docs:
            hasher.update(doc.id.encode("utf-8"))
            hasher.update(doc.text.encode("utf-8"))
        return hasher.hexdigest()[:16]

    def _cache_path(self) -> Path:
        return self._cache_dir / f"policy_embeddings_{self._content_hash()}.npz"

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        """L2-normalize rows so dot products equal cosine similarity."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).astype(np.float32)

    def _try_load_cache(self) -> np.ndarray | None:
        """Load cached embeddings if present and shaped consistently."""
        cache_path = self._cache_path()
        if not cache_path.exists():
            return None
        try:
            data = np.load(cache_path)
            matrix = data["embeddings"]
            if matrix.shape[0] == len(self._docs):
                logger.info("Loaded policy embeddings from cache %s", cache_path)
                return matrix.astype(np.float32)
        except Exception as exc:  # noqa: BLE001 - cache is best-effort
            logger.warning("Failed to load embedding cache (%s); rebuilding.", exc)
        return None

    def _save_cache(self, matrix: np.ndarray) -> None:
        """Persist embeddings to the cache (best-effort)."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(self._cache_path(), embeddings=matrix)
        except Exception as exc:  # noqa: BLE001 - cache is best-effort
            logger.warning("Failed to write embedding cache (%s).", exc)

    async def _ensure_index(self) -> None:
        """Build (or load) the normalized embedding matrix for all documents."""
        if self._matrix is not None:
            return
        cached = self._try_load_cache()
        if cached is not None:
            self._matrix = self._normalize(cached)
            return
        raw = await self._embedder.embed([doc.text for doc in self._docs])
        self._save_cache(raw)
        self._matrix = self._normalize(raw)

    async def retrieve(self, query: str, k: int = 3) -> list[PolicyDoc]:
        """Return the top-``k`` policy documents most similar to ``query``.

        Parameters
        ----------
        query : str
            The complaint text (or any query) to match against the KB.
        k : int, optional, default=3
            Maximum number of documents to return.

        Returns
        -------
        list[PolicyDoc]
            Documents sorted by descending cosine similarity, each with ``score`` set.
        """
        await self._ensure_index()
        assert self._matrix is not None  # for type-checkers; set by _ensure_index

        query_vec = self._normalize(await self._embedder.embed([query]))[0]
        scores = self._matrix @ query_vec
        top_idx = np.argsort(-scores)[: max(0, k)]
        return [self._docs[i].model_copy(update={"score": float(scores[i])}) for i in top_idx]

    async def aclose(self) -> None:
        """Close the underlying embedding client."""
        await self._embedder.aclose()
