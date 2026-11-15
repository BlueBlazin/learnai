import mlx.core as mx

from whetstone.kit import CountingExperts


def moe_sparse(x: mx.array, w_router: mx.array, experts: CountingExperts, k: int) -> mx.array:
    """A mixture-of-experts layer that runs each expert only on the tokens routed to it.

    x: float32 tensor of shape (N, D), one row per token.
    w_router: float32 tensor of shape (E, D); router logits are x @ w_router.T.
    experts: experts(e, rows) runs expert e on rows of shape (n, D) and returns (n, D);
        every call and every row is counted. experts.n_experts is E.
    k: experts per token, 1 <= k <= E.
    Returns a float32 tensor of shape (N, D): for each token, the sum over its k
    selected experts of (renormalized router probability) * (expert output).
    Each selected expert is called once, on exactly the rows routed to it.
    """
    raise NotImplementedError
