import json
import os
import time

import numpy as np
import pandas as pd

import app as core


def _dominant_freq(ts: pd.Series) -> str:
    clean = pd.Series(pd.to_datetime(ts, utc=True)).dropna().sort_values().drop_duplicates()
    if len(clean) < 3:
        return "3min"
    diffs = clean.diff().dropna()
    if diffs.empty:
        return "3min"
    try:
        dominant = diffs.mode().iloc[0]
    except Exception:
        dominant = diffs.median()
    seconds = max(1, int(round(pd.Timedelta(dominant).total_seconds())))
    if seconds % 3600 == 0:
        return f"{max(1, seconds // 3600)}h"
    if seconds % 60 == 0:
        return f"{max(1, seconds // 60)}min"
    return f"{seconds}s"


def chronos_predict_compat(df: pd.DataFrame, horizon: int):
    pipe = core.get_chronos()

    # Feed Chronos concrete NumPy-backed dtypes, timezone-naive timestamps,
    # and an explicit market frequency to avoid pandas extension-dtype edge cases.
    ts_utc = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    ts_naive = ts_utc.dt.tz_convert(None)
    target = pd.to_numeric(df["ret"], errors="coerce").astype("float64")

    context = pd.DataFrame(
        {
            "id": np.asarray(["series"] * len(df), dtype=object),
            "timestamp": np.asarray(ts_naive.to_numpy(dtype="datetime64[ns]"), dtype="datetime64[ns]"),
            "target": np.asarray(target.to_numpy(dtype=np.float64), dtype=np.float64),
        }
    ).dropna(subset=["timestamp", "target"])

    if len(context) < max(16, horizon + 3):
        raise ValueError("not enough clean observations for Chronos")

    freq = _dominant_freq(df["timestamp"])
    pred = pipe.predict_df(
        context,
        prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
        freq=freq,
    )

    med_col = "predictions" if "predictions" in pred.columns else "0.5"
    med = np.asarray(pd.to_numeric(pred[med_col], errors="coerce"), dtype=np.float64)[-horizon:]
    q10 = (
        np.asarray(pd.to_numeric(pred["0.1"], errors="coerce"), dtype=np.float64)[-horizon:]
        if "0.1" in pred.columns
        else med.copy()
    )
    q90 = (
        np.asarray(pd.to_numeric(pred["0.9"], errors="coerce"), dtype=np.float64)[-horizon:]
        if "0.9" in pred.columns
        else med.copy()
    )
    if not (np.isfinite(med).all() and np.isfinite(q10).all() and np.isfinite(q90).all()):
        raise ValueError("Chronos returned non-finite forecasts")
    return med, q10, q90


core.chronos_predict = chronos_predict_compat
app = core.app


@app.get("/selftest-compat")
def selftest_compat():
    """One fixed XAUUSD M3 inference check without exposing internal secrets."""
    token = os.getenv("INTERNAL_SERVICE_TOKEN", "").strip()
    if not token:
        result = {"ok": False, "error": "internal token unavailable"}
        print("DEEP_SELFTEST=" + json.dumps(result, separators=(",", ":")), flush=True)
        return result

    req = core.ForecastRequest(symbol="XAUUSD", timeframe="M3", candles=[], prediction_length=10)
    started = time.perf_counter()
    try:
        data = core.forecast(req, token)
        models = data.get("models") or {}
        result = {
            "ok": True,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "status": data.get("status"),
            "symbol": data.get("symbol"),
            "timeframe": data.get("timeframe"),
            "direction": data.get("direction"),
            "direction_confidence": data.get("direction_confidence"),
            "ensemble_weights": data.get("ensemble_weights"),
            "chronos_status": (models.get("chronos2") or {}).get("status"),
            "patchtst_status": (models.get("patchtst") or {}).get("status"),
            "tft_status": (models.get("tft") or {}).get("status"),
            "local_training": data.get("local_training"),
            "quality_note": data.get("quality_note"),
        }
    except Exception as exc:
        result = {
            "ok": False,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "error": f"{type(exc).__name__}: {exc}"[:700],
        }
    print("DEEP_SELFTEST=" + json.dumps(result, default=str, separators=(",", ":")), flush=True)
    return result
