# BUG: similarity is the raw dot product instead of the cosine, so long vectors score higher and the threshold means nothing for unnormalized embeddings.
from dataclasses import dataclass


def namespace_of(request: dict) -> tuple | None:
    if request.get("tools") or request.get("temperature", 1.0) != 0:
        return None
    system = "\n".join(m["content"] for m in request["messages"] if m["role"] == "system")
    return (request["model"], system, request.get("max_tokens"))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot


@dataclass(eq=False)
class Entry:
    namespace: object
    embedding: list[float]
    response: str
    stored_at: float
    order: int


class SemanticCache:
    def __init__(self, threshold: float, ttl_s: float, capacity: int):
        self.threshold = threshold
        self.ttl_s = ttl_s
        self.capacity = capacity
        self.entries: list[Entry] = []
        self.puts = 0

    def live(self, entry: Entry, now: float) -> bool:
        return now - entry.stored_at < self.ttl_s

    def get(self, embedding: list[float], now: float, namespace) -> str | None:
        best, best_key = None, None
        for entry in self.entries:
            if entry.namespace != namespace or not self.live(entry, now):
                continue
            score = cosine(embedding, entry.embedding)
            if score < self.threshold:
                continue
            key = (score, entry.order)
            if best is None or key > best_key:
                best, best_key = entry, key
        if best is None:
            return None
        self.entries.remove(best)
        self.entries.append(best)
        return best.response

    def put(self, embedding: list[float], response: str, now: float, namespace) -> None:
        self.entries = [entry for entry in self.entries if self.live(entry, now)]
        if len(self.entries) >= self.capacity:
            self.entries.pop(0)
        self.entries.append(Entry(namespace, list(embedding), response, now, self.puts))
        self.puts += 1
