# BUG: there is no tolerance band: the count is recomputed from the exact ratio at every evaluation, so load that sits just under the target of the current replicas still produces a lower recommendation.
def desired_replicas(metrics, state, cfg):
    current = metrics.ready + metrics.starting  # replicas that are starting count
    load = metrics.queue_depth + metrics.running  # requests in the house

    # The proportional rule: as many replicas as bring the load per replica to the
    # target. ceil(current * (load / current) / target) is ceil(load / target).
    raw = -(-load // cfg.target)
    raw = max(cfg.min_replicas, min(cfg.max_replicas, raw))

    # Remember this recommendation; forget those older than the window.
    history = [(t, n) for t, n in (state or ()) if metrics.now - t < cfg.window_s]
    history.append((metrics.now, raw))

    if raw >= current:
        n = raw  # scale up at once
    else:
        # Scale down only as far as every recommendation in the window agrees to.
        n = min(current, max(n for _, n in history))
    return n, history
