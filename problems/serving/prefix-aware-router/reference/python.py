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
        least = min(replica.queue_len for replica in replicas)
        # Affinity may only choose among replicas that are not much busier than the
        # idlest one: this is what stops a hot prefix from burying one replica.
        eligible = [r for r in replicas if r.queue_len - least <= self.imbalance]
        best = max(eligible, key=lambda r: (self.match(tokens, r.id), -r.queue_len, -r.id))
        if self.match(tokens, best.id) >= self.match_threshold * len(tokens):
            chosen = best
        else:
            chosen = min(replicas, key=lambda r: (r.queue_len, r.id))
        # The chosen replica will hold this prompt's keys and values: remember it, so
        # the next request with the same prefix follows.
        self.index.setdefault(chosen.id, deque(maxlen=self.index_size)).append(tuple(tokens))
        return chosen.id
