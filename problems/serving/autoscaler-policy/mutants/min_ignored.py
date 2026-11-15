# BUG: the recommendation is capped at max_replicas but not raised to min_replicas, so an idle cluster is scaled to zero and the next request finds no replica.
def desired_replicas(metrics, state, cfg):
    current = metrics.ready + metrics.starting  # replicas that are starting count
    load = metrics.queue_depth + metrics.running  # requests in the house

    # The proportional rule: as many replicas as bring the load per replica to the
    # target. ceil(current * (load / current) / target) is ceil(load / target).
    capacity = current * cfg.target
    if abs(load - capacity) <= cfg.tolerance * capacity:
        raw = current  # close enough: do not chase noise
    else:
        raw = -(-load // cfg.target)
    raw = min(cfg.max_replicas, raw)

    # Remember this recommendation; forget those older than the window.
    history = [(t, n) for t, n in (state or ()) if metrics.now - t < cfg.window_s]
    history.append((metrics.now, raw))

    if raw >= current:
        n = raw  # scale up at once
    else:
        # Scale down only as far as every recommendation in the window agrees to.
        n = min(current, max(n for _, n in history))
    return n, history
