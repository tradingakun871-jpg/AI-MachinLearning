import pandas as pd


def _trend(df, span_fast=20, span_slow=50):
    f = df.close.ewm(span=span_fast, adjust=False).mean()
    s = df.close.ewm(span=span_slow, adjust=False).mean()
    return (f>s).astype(int) - (f<s).astype(int)


def attach_htf_context(m3: pd.DataFrame, m5: pd.DataFrame, m15: pd.DataFrame) -> pd.DataFrame:
    """Attach only HTF candles that were fully closed when an M3 decision became available."""
    base = m3.copy().sort_values("timestamp")
    base["decision_time"] = pd.to_datetime(base["timestamp"], utc=True) + pd.Timedelta(minutes=3)

    for frame, name, minutes in [(m5,"trend_m5",5),(m15,"trend_m15",15)]:
        h = frame.copy().sort_values("timestamp")
        h[name] = _trend(h)
        h["available_at"] = pd.to_datetime(h["timestamp"], utc=True) + pd.Timedelta(minutes=minutes)
        h = h[["available_at", name]].sort_values("available_at")
        base = pd.merge_asof(
            base.sort_values("decision_time"),
            h,
            left_on="decision_time",
            right_on="available_at",
            direction="backward",
        )
        base = base.drop(columns=["available_at"])

    return base.drop(columns=["decision_time"])
