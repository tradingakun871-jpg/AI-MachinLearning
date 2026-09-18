def barrier_outcome(future_high, future_low, entry, side, stop_distance, rr=3.0):
    """Causal TP-before-SL outcome with diagnostics.

    The caller must pass bars strictly AFTER the signal candle. Entry should be
    the next tradable bar price (the dataset builder uses next-bar open).
    Ambiguous same-bar TP+SL is unresolved rather than guessed.
    """
    side=side.upper()
    if side not in ("BUY","SELL"):
        raise ValueError("side must be BUY or SELL")
    if stop_distance is None or float(stop_distance) <= 0:
        return {"label":None,"reason":"INVALID_STOP"}

    sign=1 if side=="BUY" else -1
    entry=float(entry)
    stop_distance=float(stop_distance)
    tp=entry + sign*stop_distance*float(rr)
    sl=entry - sign*stop_distance

    for bars,(high,low) in enumerate(zip(future_high,future_low),start=1):
        high=float(high); low=float(low)
        hit_tp=high>=tp if sign>0 else low<=tp
        hit_sl=low<=sl if sign>0 else high>=sl
        if hit_tp and hit_sl:
            return {
                "label":None,"reason":"AMBIGUOUS_SAME_BAR","entry":entry,
                "sl":sl,"tp":tp,"bars_to_outcome":bars,
            }
        if hit_tp:
            return {
                "label":1,"reason":"TP","entry":entry,
                "sl":sl,"tp":tp,"bars_to_outcome":bars,
            }
        if hit_sl:
            return {
                "label":0,"reason":"SL","entry":entry,
                "sl":sl,"tp":tp,"bars_to_outcome":bars,
            }

    return {
        "label":None,"reason":"UNRESOLVED","entry":entry,
        "sl":sl,"tp":tp,"bars_to_outcome":None,
    }


def triple_barrier_label(future_high, future_low, entry, side, stop_distance, rr=3.0):
    """Backward-compatible label-only wrapper."""
    return barrier_outcome(
        future_high,future_low,entry,side,stop_distance,rr
    )["label"]
