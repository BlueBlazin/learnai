class Router:
    """Routes each request to the replica most likely to hold its prompt's prefix in
    its KV cache, as long as that replica is not much busier than the least loaded."""

    def __init__(self, match_threshold: float, imbalance: int, index_size: int):
        """match_threshold: the smallest fraction of a prompt that must match for
            affinity to be used, in [0, 1].
        imbalance: how many more requests than the least loaded replica a replica may
            hold and still be chosen for its match.
        index_size: how many prompts the router remembers per replica.
        """
        raise NotImplementedError

    def route(self, tokens: tuple[int, ...], replicas: list) -> int:
        """Choose a replica for a prompt with these token ids.

        replicas: the candidates, each with `.id` (int) and `.queue_len` (int).
        Returns the id of the chosen replica.
        """
        raise NotImplementedError
