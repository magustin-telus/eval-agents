"""Embedding client for the complaint-resolution policy knowledge base.

This module provides a small async wrapper around an OpenAI-compatible
``/embeddings`` endpoint, used to embed policy documents and complaint queries
for the in-process vector store in :mod:`aieng.agent_evals.complaint_resolution.kb`.

Endpoint resolution
-------------------
The client prefers a dedicated embedding service when one is configured via
``Configs.embedding_base_url`` / ``Configs.embedding_api_key`` (e.g. a hosted
``@cf/baai/bge-m3`` deployment). When that is not configured, it falls back to
the same OpenAI-compatible endpoint used for chat (``Configs.openai_base_url`` /
``Configs.openai_api_key``, i.e. the Gemini endpoint) with a Gemini embedding
model. This keeps the module working in environments that only have the standard
Gemini API key set, while still honoring a dedicated embedding service when present.
"""

import logging

import numpy as np
from aieng.agent_evals.configs import Configs
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


logger = logging.getLogger(__name__)

# Default embedding model for the OpenAI-compatible Gemini endpoint fallback.
DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

# Max number of inputs per embeddings request. The policy KB is tiny, but
# complaint batches during dataset embedding can be larger, so we chunk.
_EMBED_BATCH_SIZE = 64

_RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)


class EmbeddingClient:
    """Async embedding client over an OpenAI-compatible endpoint.

    Parameters
    ----------
    config : Configs | None, optional
        Configuration object. If ``None``, a default ``Configs()`` is created.

    Examples
    --------
    >>> client = EmbeddingClient()
    >>> vecs = await client.embed(["my credit card was charged twice"])
    >>> vecs.shape[0]
    1
    >>> await client.aclose()
    """

    def __init__(self, config: Configs | None = None) -> None:
        self._config = config or Configs()  # type: ignore[call-arg]
        self._client: AsyncOpenAI | None = None
        self._model: str = self._resolve_model()

    def _resolve_model(self) -> str:
        """Pick the embedding model based on which endpoint is configured."""
        cfg = self._config
        if cfg.embedding_base_url and cfg.embedding_api_key:
            return cfg.embedding_model_name
        return DEFAULT_GEMINI_EMBEDDING_MODEL

    @property
    def model(self) -> str:
        """Name of the embedding model in use."""
        return self._model

    @property
    def client(self) -> AsyncOpenAI:
        """Get or create the underlying async client (lazy).

        Uses the dedicated embedding endpoint when configured, otherwise the
        standard OpenAI-compatible (Gemini) endpoint.
        """
        if self._client is None:
            cfg = self._config
            if cfg.embedding_base_url and cfg.embedding_api_key:
                base_url = cfg.embedding_base_url
                api_key = cfg.embedding_api_key.get_secret_value()
                logger.info("Embedding via dedicated endpoint (model=%s)", self._model)
            else:
                base_url = cfg.openai_base_url
                api_key = cfg.openai_api_key.get_secret_value()
                logger.info("Embedding via OpenAI-compatible endpoint (model=%s)", self._model)
            # Custom retry handled by tenacity below, so disable client retries.
            self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        return self._client

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch with retry on transient API errors."""
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=2.0, min=1.0, max=30.0),
            retry=retry_if_exception_type(_RETRYABLE_ERRORS),
            reraise=True,
        ):
            with attempt:
                response = await self.client.embeddings.create(model=self._model, input=texts)
                return [item.embedding for item in response.data]
        raise RuntimeError("Embedding call failed unexpectedly without a result.")

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into a 2-D float32 array.

        Parameters
        ----------
        texts : list[str]
            Texts to embed. Must be non-empty.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(texts), embedding_dim)`` and dtype ``float32``.

        Raises
        ------
        ValueError
            If ``texts`` is empty.
        """
        if not texts:
            raise ValueError("`texts` must contain at least one string to embed.")

        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[start : start + _EMBED_BATCH_SIZE]
            vectors.extend(await self._embed_batch(batch))

        return np.asarray(vectors, dtype=np.float32)

    async def aclose(self) -> None:
        """Close the underlying async client if it was created."""
        if self._client is not None:
            await self._client.close()
            self._client = None
