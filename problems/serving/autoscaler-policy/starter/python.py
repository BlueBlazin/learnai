def desired_replicas(metrics, state, cfg):
    """One evaluation of the autoscaling policy.

    metrics: the cluster now: `.now` (seconds), `.ready`, `.starting` (replicas),
        `.queue_depth` (requests waiting), `.running` (requests being served).
    state: None on the first call, afterwards whatever the previous call returned.
    cfg: `.target` (requests in flight per replica), `.tolerance`, `.min_replicas`,
        `.max_replicas`, `.window_s` (scale-down stabilization window, seconds).
    Returns (n, state): the number of replicas that should be starting or ready,
    and the state to pass to the next call.
    """
    raise NotImplementedError
