import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from data.feature_builder import add_market_features
from ml.features import FEATURE_COLUMNS
from ml.regime import RegimeClassifier
from ml.setup_features import add_setup_features
from smc.engine import add_smc_features


def synthetic_market(rows=600):
    rng = np.random.default_rng(42)
    ret = rng.normal(0.00002, 0.0007, rows)
    close = 2300.0 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.maximum(close * 0.00003, 0.01)
    excursion = np.maximum(np.abs(close - open_), close * rng.uniform(0.0001, 0.0008, rows))
    high = np.maximum(open_, close) + excursion * rng.uniform(0.2, 1.0, rows)
    low = np.minimum(open_, close) - excursion * rng.uniform(0.2, 1.0, rows)
    volume = rng.lognormal(mean=8.0, sigma=0.55, size=rows)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="3min", tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": volume,
            "spread": spread,
        }
    )


def main():
    d = add_market_features(synthetic_market())
    d = add_smc_features(d)
    d["trend_m3"] = d["structure_bias"]
    d["trend_m5"] = d["structure_bias"].rolling(2, min_periods=1).mean().apply(np.sign)
    d["trend_m15"] = d["structure_bias"].rolling(5, min_periods=1).mean().apply(np.sign)
    d = add_setup_features(d)
    d["stop_distance_atr"] = 1.0

    missing = [name for name in FEATURE_COLUMNS if name not in d.columns]
    assert not missing, f"missing features: {missing}"
    assert set(d["pa_direction"].dropna().unique()).issubset({-1, 0, 1})
    assert set(d["orderflow_direction"].dropna().unique()).issubset({-1, 0, 1})
    assert float(d["orderflow_available"].iloc[-1]) == 1.0
    assert float(d["orderflow_is_proxy"].iloc[-1]) == 1.0
    assert float(d["orderflow_source_quality"].iloc[-1]) > 0.0

    clean = d.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train = clean.iloc[:400].copy()
    test = clean.iloc[400:].copy()
    regime = RegimeClassifier().fit(train)
    pred = regime.predict(test)
    assert len(pred) == len(test)
    assert np.isfinite(clean[FEATURE_COLUMNS].to_numpy(dtype=float)).all()

    print(
        {
            "ok": True,
            "rows": len(d),
            "feature_count": len(FEATURE_COLUMNS),
            "price_action": {
                "directional_bars": int(d["pa_direction"].ne(0).sum()),
                "breakouts": int(d["pa_breakout_direction"].ne(0).sum()),
                "rejections": int(d["pa_rejection_direction"].ne(0).sum()),
            },
            "orderflow": {
                "available": bool(d["orderflow_available"].iloc[-1]),
                "proxy": bool(d["orderflow_is_proxy"].iloc[-1]),
                "source_quality": float(d["orderflow_source_quality"].iloc[-1]),
                "directional_bars": int(d["orderflow_direction"].ne(0).sum()),
            },
            "regimes": sorted(set(int(x) for x in pred)),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
