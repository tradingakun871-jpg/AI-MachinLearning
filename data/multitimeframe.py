import pandas as pd


def _trend(df, span_fast=20, span_slow=50):
    f = df.close.ewm(span=span_fast, adjust=False).mean()
    s = df.close.ewm(span=span_slow, adjust=False).mean()
    return (f>s).astype(int) - (f<s).astype(int)


def attach_htf_context(m3: pd.DataFrame, m5: pd.DataFrame, m15: pd.DataFrame) -> pd.DataFrame:
    """Backward as-of joins prevent future HTF candles leaking into M3 rows."""
    base = m3.copy().sort_values("timestamp")
    for frame, name in [(m5,"trend_m5"),(m15,"trend_m15")]:
        h = frame.copy().sort_values("timestamp")
        h[name] = _trend(h)
        h = h[["timestamp", name]]
        base = pd.merge_asof(base, h, on="timestamp", direction="backward")
    return base
