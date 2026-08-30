"""Embedder protocol and implementations.

Separate from LLMClient even though litellm can serve both, because the two change on
different clocks: a completion model can be swapped between wakes with no consequence;
an embedding model cannot be swapped at all without `rebuild --from-history`. That is
why `model_id` and `dim` are on the contract — they are what a stored vector records so
a mismatch is detectable instead of silently returning bad neighbors.

Only this module and llm.client may import litellm or name a model.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import time
from typing import Protocol, runtime_checkable

Vector = list[float]

# The pinned defaults, overridable through Settings. Kept here so no module outside
# llm/ names a model (invariant 20).
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
DEFAULT_EMBEDDING_DIM = 1536


@runtime_checkable
class Embedder(Protocol):
    model_id: str
    dim: int

    def embed(self, texts: list[str]) -> list[Vector]: ...


class LiteLLMEmbedder:
    """Hosted embeddings through litellm. Batches calls; retries with jitter."""

    def __init__(
        self,
        model: str,
        dim: int,
        *,
        batch_size: int = 96,
        timeout: float = 30.0,
        retries: int = 3,
    ) -> None:
        self.model_id = model
        self.dim = dim
        self._batch_size = batch_size
        self._timeout = timeout
        self._retries = retries

    def embed(self, texts: list[str]) -> list[Vector]:
        import litellm

        out: list[Vector] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = self._with_retries(litellm, batch)
            rows = sorted(response.data, key=lambda d: d["index"])
            out.extend([list(map(float, row["embedding"])) for row in rows])
        for vector in out:
            if len(vector) != self.dim:
                raise ValueError(f"{self.model_id} returned dim {len(vector)}, configured dim is {self.dim}")
        return out

    def _with_retries(self, litellm, batch: list[str]):
        delay = 1.0
        for attempt in range(self._retries):
            # Most-specific-first: retryable classes are caught; anything else
            # (auth, bad request, APIStatusError) propagates immediately.
            try:
                return litellm.embedding(model=self.model_id, input=batch, timeout=self._timeout)
            except (
                litellm.exceptions.RateLimitError,
                litellm.exceptions.APIConnectionError,
                litellm.exceptions.Timeout,
            ):
                if attempt == self._retries - 1:
                    raise
                time.sleep(delay + random.random())
                delay *= 2
        raise AssertionError("unreachable")


class HashEmbedder:
    """Deterministic, network-free embedder: a feature-hashed bag of words, L2-normalized,
    so cosine similarity tracks token overlap. For tests and offline eval runs — it is
    not a semantic model, and its vectors are never comparable with a real embedder's
    (the recorded model_id is what makes that mismatch visible)."""

    def __init__(self, dim: int = 256) -> None:
        self.model_id = f"fake/hash-bow-{dim}"
        self.dim = dim

    def embed(self, texts: list[str]) -> list[Vector]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> Vector:
        vector = [0.0] * self.dim
        for token in re.findall(r"[\w']+", text.lower()):
            h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest())
            sign = 1.0 if (h >> 63) & 1 == 0 else -1.0
            vector[h % self.dim] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]
