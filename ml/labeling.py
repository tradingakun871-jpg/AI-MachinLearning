def triple_barrier_label(future_high, future_low, entry, side, stop_distance, rr=3.0):
    """Return 1 when TP is touched before SL, 0 when SL is touched first, None if unresolved.
    Ambiguous same-bar TP+SL is deliberately unresolved to avoid optimistic bias.
    """
    sign = 1 if side.upper() == "BUY" else -1
    tp = entry + sign * stop_distance * rr
    sl = entry - sign * stop_distance
    for high, low in zip(future_high, future_low):
        hit_tp = high >= tp if sign > 0 else low <= tp
        hit_sl = low <= sl if sign > 0 else high >= sl
        if hit_tp and hit_sl:
            return None
        if hit_tp:
            return 1
        if hit_sl:
            return 0
    return None
