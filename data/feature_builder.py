import numpy as np
import pandas as pd


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_values("timestamp").reset_index(drop=True)
    prev_close = d["close"].shift(1)
    tr = pd.concat([(d.high-d.low).abs(), (d.high-prev_close).abs(), (d.low-prev_close).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14, min_periods=14).mean()
    d["atr_norm"] = d.atr / d.close.replace(0, np.nan)
    d["return_1"] = d.close.pct_change(1)
    d["return_3"] = d.close.pct_change(3)
    d["range_atr"] = (d.high-d.low) / d.atr.replace(0, np.nan)
    vol = d["return_1"].rolling(50).std()
    d["volatility_z"] = (vol-vol.rolling(200).mean()) / vol.rolling(200).std().replace(0, np.nan)
    if "spread" not in d: d["spread"] = 0.0
    d["spread_atr"] = d.spread / d.atr.replace(0, np.nan)
    ts = pd.to_datetime(d.timestamp, utc=True)
    hour = ts.dt.hour + ts.dt.minute/60
    d["hour_sin"] = np.sin(2*np.pi*hour/24)
    d["hour_cos"] = np.cos(2*np.pi*hour/24)
    return d


def add_structure_proxy(df: pd.DataFrame, fast=20, slow=50) -> pd.DataFrame:
    """Causal structure baseline. Later versions can replace this with full BOS/MSS/OB/FVG detection."""
    d = df.copy()
    ema_fast = d.close.ewm(span=fast, adjust=False).mean()
    ema_slow = d.close.ewm(span=slow, adjust=False).mean()
    d["trend_m3"] = np.sign(ema_fast-ema_slow).fillna(0)
    swing_hi = d.high.shift(1).rolling(10).max()
    swing_lo = d.low.shift(1).rolling(10).min()
    d["mss"] = np.where(d.close > swing_hi, 1, np.where(d.close < swing_lo, -1, 0))
    d["liquidity_sweep"] = (((d.low < swing_lo) & (d.close > swing_lo)) | ((d.high > swing_hi) & (d.close < swing_hi))).astype(int)
    d["order_block"] = 0
    d["fvg"] = (((d.low > d.high.shift(2)) | (d.high < d.low.shift(2)))).astype(int)
    return d
