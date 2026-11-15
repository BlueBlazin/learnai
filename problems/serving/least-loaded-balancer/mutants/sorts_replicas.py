# BUG: the replica ids are sorted before they are indexed, so the uniforms select different replicas than specified whenever the list is not in ascending order.
def pick(replicas: list[int], load: dict[int, int], u1: float, u2: float) -> int:
    n = len(replicas)
    if n == 1:
        return replicas[0]
    i = int(u1 * n)  # the first choice: uniform over all n
    j = int(u2 * (n - 1))  # the second: uniform over the other n - 1
    if j >= i:
        j += 1  # skip over the first choice, so the two are always distinct
    ordered = sorted(replicas)
    first, second = ordered[i], ordered[j]
    # Strictly fewer tokens wins; on a tie the first draw stands, which is fair
    # because either replica of a tied pair is equally likely to be drawn first.
    return second if load[second] < load[first] else first


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
