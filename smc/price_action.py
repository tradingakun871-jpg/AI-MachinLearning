import numpy as np
import pandas as pd


def add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add causal price-action features using current and past bars only."""
    d = df.copy()
    o = pd.to_numeric(d["open"], errors="coerce")
    h = pd.to_numeric(d["high"], errors="coerce")
    l = pd.to_numeric(d["low"], errors="coerce")
    c = pd.to_numeric(d["close"], errors="coerce")

    rng = (h - l).abs().replace(0, np.nan)
    body = c - o
    abs_body = body.abs()
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l

    d["pa_body_ratio"] = (abs_body / rng).clip(0, 1).fillna(0.0)
    d["pa_upper_wick_ratio"] = (upper / rng).clip(0, 1).fillna(0.0)
    d["pa_lower_wick_ratio"] = (lower / rng).clip(0, 1).fillna(0.0)
    d["pa_close_location"] = ((c - l) / rng).clip(0, 1).fillna(0.5)
    d["pa_candle_direction"] = np.sign(body).fillna(0).astype(int)

    po, pc = o.shift(1), c.shift(1)
    bullish_engulf = (body > 0) & (pc < po) & (o <= pc) & (c >= po)
    bearish_engulf = (body < 0) & (pc > po) & (o >= pc) & (c <= po)
    d["pa_engulfing_direction"] = np.where(
        bullish_engulf, 1, np.where(bearish_engulf, -1, 0)
    ).astype(int)

    ph, pl = h.shift(1), l.shift(1)
    d["pa_inside_bar"] = ((h <= ph) & (l >= pl)).astype(int)
    d["pa_outside_bar"] = ((h >= ph) & (l <= pl)).astype(int)

    bullish_rejection = (
        (d["pa_lower_wick_ratio"] >= 0.50)
        & (d["pa_close_location"] >= 0.60)
        & (d["pa_body_ratio"] <= 0.45)
    )
    bearish_rejection = (
        (d["pa_upper_wick_ratio"] >= 0.50)
        & (d["pa_close_location"] <= 0.40)
        & (d["pa_body_ratio"] <= 0.45)
    )
    d["pa_rejection_direction"] = np.where(
        bullish_rejection, 1, np.where(bearish_rejection, -1, 0)
    ).astype(int)

    prior_high = h.shift(1).rolling(20, min_periods=8).max()
    prior_low = l.shift(1).rolling(20, min_periods=8).min()
    d["pa_breakout_direction"] = np.where(
        c > prior_high, 1, np.where(c < prior_low, -1, 0)
    ).astype(int)

    range_atr = pd.to_numeric(d.get("range_atr", 0.0), errors="coerce").fillna(0.0)
    displacement = (d["pa_body_ratio"] >= 0.65) & (range_atr >= 1.20)
    d["pa_displacement_direction"] = np.where(
        displacement, d["pa_candle_direction"], 0
    ).astype(int)

    atr = pd.to_numeric(d.get("atr", np.nan), errors="coerce").replace(0, np.nan)
    d["pa_momentum_3_atr"] = ((c - c.shift(3)) / atr).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    path = c.diff().abs().rolling(10, min_periods=5).sum()
    direct = (c - c.shift(10)).abs()
    d["pa_trend_efficiency"] = (direct / path.replace(0, np.nan)).clip(0, 1).fillna(0.0)

    short_range = rng.rolling(5, min_periods=3).mean()
    long_range = rng.rolling(20, min_periods=8).mean()
    compression_ratio = short_range / long_range.replace(0, np.nan)
    d["pa_compression"] = (1.0 - compression_ratio).clip(-1, 1).fillna(0.0)

    components = pd.DataFrame(
        {
            "engulf": d["pa_engulfing_direction"],
            "reject": d["pa_rejection_direction"],
            "breakout": d["pa_breakout_direction"],
            "displace": d["pa_displacement_direction"],
            "candle": d["pa_candle_direction"] * d["pa_body_ratio"],
        },
        index=d.index,
    )
    score = components.sum(axis=1)
    # Ignore weak candle noise. Direction only becomes context when a strong
    # body or one of the named price-action events creates meaningful evidence.
    d["pa_direction"] = np.where(
        score >= 0.35, 1, np.where(score <= -0.35, -1, 0)
    ).astype(int)
    d["pa_strength"] = (components.abs().sum(axis=1) / 5.0).clip(0, 1).fillna(0.0)
    return d
