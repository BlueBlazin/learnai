import jax
import jax.numpy as jnp


def search(
    queries: jax.Array, corpus: jax.Array, k: int, metric: str = "cosine"
) -> tuple[jax.Array, jax.Array]:
    if metric == "cosine":
        # Cosine similarity is the inner product of unit vectors: normalize both sides.
        queries = queries / jnp.linalg.norm(queries, axis=1, keepdims=True)
        corpus = corpus / jnp.linalg.norm(corpus, axis=1, keepdims=True)
    scores = queries @ corpus.T  # (Q, N): every query against every vector, one product
    k = min(k, corpus.shape[0])

    # lax.top_k selects the k best of a row without sorting it, returns them best
    # first, and is stable by default (is_stable=True): among equal scores the lower
    # index comes first, which is the tie rule of the problem.
    values, index = jax.lax.top_k(scores, k)
    return index, values
