import jax
import jax.numpy as jnp
import numpy as np

from whetstone.kit import CountingExperts, SimDevices


def ep_forward(
    x: jax.Array,
    home: list[int],
    w_router: jax.Array,
    experts: CountingExperts,
    placement: list[int],
    devs: SimDevices,
    k: int,
) -> jax.Array:
    # Route as in moe-sparse-dispatch (the sort is stable: ties go to the lowest index).
    logits = x @ w_router.T  # (N, E)
    chosen = jnp.argsort(-logits, axis=-1)[:, :k]  # (N, k)
    weights = jax.nn.softmax(jnp.take_along_axis(logits, chosen, axis=-1), axis=-1)  # (N, k)

    # Bookkeeping on the host: which device each of a token's experts lives on.
    routes = np.asarray(chosen)
    homes, placed = np.asarray(home), np.asarray(placement)
    target = placed[routes]  # (N, k)

    y = jnp.zeros_like(x)
    for dst in range(devs.n_devices):  # the device whose experts work in this round
        # Dispatch: from every device, the rows of its tokens that need an expert on
        # dst. A token appears once, however many of its experts live there.
        batches = []
        for src in range(devs.n_devices):
            tokens = np.flatnonzero((homes == src) & (target == dst).any(axis=1))
            if tokens.size == 0:
                continue  # nothing to send: no message
            rows = x[jnp.asarray(tokens)]
            batches.append((src, tokens, rows if src == dst else devs.send(src, dst, rows)))
        if not batches:
            continue
        rows = jnp.concatenate([batch[2] for batch in batches])  # everything now on dst
        tokens = np.concatenate([batch[1] for batch in batches])  # the token of each row

        # Compute on dst: each of its experts once, on its rows from every source,
        # weighted and summed per token.
        partial = jnp.zeros_like(rows)
        for expert in np.flatnonzero(placed == dst).tolist():
            at, slot = np.nonzero(routes[tokens] == expert)  # rows on dst, and their top-k slot
            if at.size == 0:
                continue
            weight = weights[jnp.asarray(tokens[at]), jnp.asarray(slot)]
            at = jnp.asarray(at)
            partial = partial.at[at].add(weight[:, None] * experts(expert, rows[at]))

        # Combine: one message back to every source, a row per row it sent.
        start = 0
        for src, sent, _ in batches:
            part = partial[start : start + sent.size]
            start += sent.size
            if src != dst:
                part = devs.send(dst, src, part)
            y = y.at[jnp.asarray(sent)].add(part)
    return y
