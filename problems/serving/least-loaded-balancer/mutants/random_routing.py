# BUG: the load is never consulted: the first draw is returned, which is uniform random routing.
def pick(replicas: list[int], load: dict[int, int], u1: float, u2: float) -> int:
    n = len(replicas)
    if n == 1:
        return replicas[0]
    i = int(u1 * n)  # the first choice: uniform over all n
    return replicas[i]


class Balancer:
    def __init__(self, replicas: list[int]):
        self.replicas = list(replicas)
        self.estimate = dict.fromkeys(self.replicas, 0)

    def report(self, loads: dict[int, int]) -> None:
        # A report is the truth as of now: it already contains whatever this balancer
        # sent earlier and is still outstanding, so it replaces the estimate.
        self.estimate = {replica: loads[replica] for replica in self.replicas}

    def route(self, tokens: int, u1: float, u2: float) -> int:
        chosen = pick(self.replicas, self.estimate, u1, u2)
        # Until the next report, nobody else will tell us that this replica got busier.
        self.estimate[chosen] += tokens
        return chosen
