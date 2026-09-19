import numpy as np
import pandas as pd


def _numeric(frame: pd.DataFrame, name: str, default=0.0) -> pd.Series:
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").fillna(default)


def _volume_source(d: pd.DataFrame):
    """Return volume series, quality score, and whether source is only a proxy."""
    if "real_volume" in d and _numeric(d, "real_volume").gt(0).any():
        return _numeric(d, "real_volume"), 0.90, False
    if "volume" in d and _numeric(d, "volume").gt(0).any():
        return _numeric(d, "volume"), 0.75, False
    if "tick_volume" in d and _numeric(d, "tick_volume").gt(0).any():
        return _numeric(d, "tick_volume"), 0.45, True
    return pd.Series(0.0, index=d.index), 0.0, True


def add_orderflow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add causal order-flow features.

    True buy/sell volume or volume_delta is used when supplied. Otherwise the
    function creates explicitly-marked OHLCV/tick-volume proxies. Proxy order
    flow is context only and is not equivalent to exchange bid/ask delta.
    """
    d = df.copy()
    h = _numeric(d, "high")
    l = _numeric(d, "low")
    c = _numeric(d, "close")
    rng = (h - l).abs().replace(0, np.nan)
    volume, source_quality, tick_proxy = _volume_source(d)

    has_buy_sell = (
        "buy_volume" in d
        and "sell_volume" in d
        and (_numeric(d, "buy_volume") + _numeric(d, "sell_volume")).gt(0).any()
    )
    has_delta = "volume_delta" in d and _numeric(d, "volume_delta").abs().gt(0).any()

    if has_buy_sell:
        buy = _numeric(d, "buy_volume")
        sell = _numeric(d, "sell_volume")
        total = (buy + sell).replace(0, np.nan)
        delta_raw = buy - sell
        delta_ratio = (delta_raw / total).clip(-1, 1).fillna(0.0)
        volume = (buy + sell).fillna(volume)
        source_quality = 1.0
        is_proxy = 0
    elif has_delta:
        delta_raw = _numeric(d, "volume_delta")
        scale = volume.rolling(30, min_periods=8).mean().replace(0, np.nan)
        delta_ratio = (delta_raw / scale).clip(-1, 1).fillna(0.0)
        source_quality = max(source_quality, 0.95)
        is_proxy = 0
    else:
        close_location_value = ((2.0 * c - h - l) / rng).clip(-1, 1).fillna(0.0)
        delta_raw = close_location_value * volume
        scale = volume.rolling(30, min_periods=8).mean().replace(0, np.nan)
        delta_ratio = (delta_raw / scale).clip(-1, 1).fillna(0.0)
        is_proxy = 1

    available = int(volume.gt(0).any() or has_buy_sell or has_delta)
    d["orderflow_available"] = float(available)
    d["orderflow_is_proxy"] = float(is_proxy if available else 1)
    d["orderflow_source_quality"] = float(source_quality if available else 0.0)

    vol_mean = volume.rolling(50, min_periods=10).mean()
    vol_std = volume.rolling(50, min_periods=10).std().replace(0, np.nan)
    d["of_volume_z"] = ((volume - vol_mean) / vol_std).clip(-5, 5).fillna(0.0)
    d["of_delta_ratio"] = delta_ratio

    rolling_delta = delta_raw.rolling(20, min_periods=5).sum()
    rolling_volume = volume.rolling(20, min_periods=5).sum().replace(0, np.nan)
    d["of_cvd_20"] = (rolling_delta / rolling_volume).clip(-1, 1).fillna(0.0)

    range_atr = _numeric(d, "range_atr")
    rejection = _numeric(d, "pa_rejection_direction")
    baseline_range = range_atr.rolling(30, min_periods=8).median()
    absorption = (
        (d["of_volume_z"] >= 1.25)
        & (range_atr <= baseline_range.fillna(range_atr))
        & rejection.ne(0)
    )
    d["of_absorption_direction"] = np.where(absorption, rejection, 0).astype(int)

    delta_dir = np.where(
        d["of_delta_ratio"] >= 0.20,
        1,
        np.where(d["of_delta_ratio"] <= -0.20, -1, 0),
    )
    cvd_dir = np.where(
        d["of_cvd_20"] >= 0.10,
        1,
        np.where(d["of_cvd_20"] <= -0.10, -1, 0),
    )
    score = (
        0.45 * np.asarray(delta_dir, dtype=float)
        + 0.35 * np.asarray(cvd_dir, dtype=float)
        + 0.20 * d["of_absorption_direction"].to_numpy(dtype=float)
    )
    score *= d["orderflow_source_quality"].to_numpy(dtype=float)
    d["orderflow_direction"] = np.where(
        score > 0.12, 1, np.where(score < -0.12, -1, 0)
    ).astype(int)
    d["orderflow_strength"] = np.abs(score).clip(0, 1)
    return d
