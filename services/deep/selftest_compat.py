import os
import time

from compat_app import app
from app import ForecastRequest, forecast


@app.get("/selftest-compat")
def selftest_compat():
    token = os.getenv("INTERNAL_SERVICE_TOKEN", "").strip()
    if not token:
        return {"ok": False, "error": "internal token unavailable"}

    req = ForecastRequest(symbol="XAUUSD", timeframe="M3", candles=[], prediction_length=10)
    started = time.perf_counter()
    try:
        data = forecast(req, token)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        models = data.get("models") or {}
        return {
            "ok": True,
            "latency_ms": elapsed_ms,
            "status": data.get("status"),
            "symbol": data.get("symbol"),
            "timeframe": data.get("timeframe"),
            "direction": data.get("direction"),
            "direction_confidence": data.get("direction_confidence"),
            "ensemble_weights": data.get("ensemble_weights"),
            "chronos_status": (models.get("chronos2") or {}).get("status"),
            "patchtst_status": (models.get("patchtst") or {}).get("status"),
            "tft_status": (models.get("tft") or {}).get("status"),
            "training_rows": data.get("training_rows"),
            "quality_note": data.get("quality_note"),
        }
    except Exception as exc:
        return {
            "ok": False,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "error": f"{type(exc).__name__}: {exc}"[:700],
        }
