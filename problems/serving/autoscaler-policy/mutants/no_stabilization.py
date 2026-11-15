# BUG: there is no scale-down window: the replica count follows every dip in the load, so a bursty load removes replicas it needs again seconds later and each burst waits for a cold start.
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
    history = [(t, n) for t, n in (state or ()) if metrics.now - t < cfg.window_s]
    history.append((metrics.now, raw))

    return raw, history
