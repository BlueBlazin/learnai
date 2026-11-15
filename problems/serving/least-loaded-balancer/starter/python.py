def pick(replicas: list[int], load: dict[int, int], u1: float, u2: float) -> int:
    """Power of two choices: draw two distinct replicas, return the less loaded one.

    replicas: the candidate replica ids, at least one, in the order to index them.
    load: maps each replica id to its outstanding tokens.
    u1, u2: independent uniform numbers in [0, 1). u1 selects the first replica among
        all n, u2 the second among the other n - 1.
    Returns the id with strictly fewer outstanding tokens; on a tie, the first drawn.
    """
    raise NotImplementedError


class Balancer:
    """Routes requests by power of two choices on load information that may be stale."""

    def __init__(self, replicas: list[int]):
        """replicas: the replica ids, in the order to index them. Estimates start at 0."""
        raise NotImplementedError

    def report(self, loads: dict[int, int]) -> None:
        """A load report arrived: loads[id] is each replica's outstanding tokens now."""
        raise NotImplementedError

    def route(self, tokens: int, u1: float, u2: float) -> int:
        """Choose a replica for a request of `tokens` tokens and return its id."""
        raise NotImplementedError
