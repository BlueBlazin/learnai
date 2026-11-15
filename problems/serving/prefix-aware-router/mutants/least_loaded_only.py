# BUG: the index is maintained but never consulted: every request goes to the least loaded replica, which spreads each prefix over all replicas.
from collections import deque


def common_prefix(a, b) -> int:
    """The number of leading tokens two prompts share."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


class Router:
    def __init__(self, match_threshold: float, imbalance: int, index_size: int):
        self.match_threshold = match_threshold
        self.imbalance = imbalance
        self.index_size = index_size
        self.index = {}  # replica id -> the last `index_size` prompts routed to it

    def match(self, tokens, replica_id) -> int:
        """Longest prefix of `tokens` that some prompt routed to the replica shares."""
        prompts = self.index.get(replica_id, ())
        return max((common_prefix(tokens, prompt) for prompt in prompts), default=0)

    def route(self, tokens, replicas) -> int:
        chosen = min(replicas, key=lambda r: (r.queue_len, r.id))
        # The chosen replica will hold this prompt's keys and values: remember it, so
        # the next request with the same prefix follows.
        self.index.setdefault(chosen.id, deque(maxlen=self.index_size)).append(tuple(tokens))
        return chosen.id
