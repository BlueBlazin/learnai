# BUG: old recommendations are never dropped, so the highest recommendation ever made holds the replica count up forever.
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
    raw = max(cfg.min_replicas, min(cfg.max_replicas, raw))

    # Remember this recommendation; forget those older than the window.
    history = list(state or ())
    history.append((metrics.now, raw))

    if raw >= current:
        n = raw  # scale up at once
    else:
        # Scale down only as far as every recommendation in the window agrees to.
        n = min(current, max(n for _, n in history))
    return n, history
