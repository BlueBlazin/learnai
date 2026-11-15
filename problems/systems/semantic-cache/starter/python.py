import math


def namespace_of(request: dict) -> tuple | None:
    """The part of a chat request, other than the question, that shapes the response.

    request: {"model": str, "messages": [{"role", "content"}, ...]} and optionally
        "temperature" (absent: 1.0), "max_tokens" (absent: None), "tools".

    Returns None if the request uses tools or has a temperature other than 0;
    otherwise (model, system, max_tokens), where system is the contents of the system
    messages joined with "\n".
    """
    raise NotImplementedError


class SemanticCache:
    """Responses stored under embeddings, looked up by cosine similarity."""

    def __init__(self, threshold: float, ttl_s: float, capacity: int):
        """threshold: the lowest similarity that counts as a match.
        ttl_s: how long an entry lives after it is stored.
        capacity: the most entries the cache holds, over all namespaces.
        """
        raise NotImplementedError

    def get(self, embedding: list[float], now: float, namespace) -> str | None:
        """The response of the most similar live entry of `namespace` whose similarity
        is at least the threshold (ties: the most recently stored), or None."""
        raise NotImplementedError

    def put(self, embedding: list[float], response: str, now: float, namespace) -> None:
        """Drop expired entries, evict the least recently used entry if the cache is
        full, and store the new entry."""
        raise NotImplementedError
