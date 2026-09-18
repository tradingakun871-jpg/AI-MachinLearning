import os
import time

from app import app, ForecastRequest, forecast


@app.get("/selftest")
def selftest():
    token = os.getenv("INTERNAL_SERVICE_TOKEN", "").strip()
    if not token:
        return {"ok": False, "error": "internal token unavailable"}

    req = ForecastRequest(
        symbol="XAUUSD",
        timeframe="M3",
        candles=[],
        prediction_length=10,
    )
    started = time.perf_counter()
    try:
        data = forecast(req, token)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        models = data.get("models") or {}
        local = data.get("local_models") or data.get("local_training") or {}
        return {
            "ok": True,
            "http_selftest": 200,
            "latency_ms": elapsed_ms,
            "status": data.get("status"),
            "symbol": data.get("symbol"),
            "timeframe": data.get("timeframe"),
            "direction": data.get("direction"),
            "direction_confidence": data.get("direction_confidence"),
            "ensemble_weights": data.get("ensemble_weights"),
            "chronos_present": "chronos2" in models,
            "patchtst_present": "patchtst" in models,
            "tft_present": "tft" in models,
            "local_state": local.get("state") if isinstance(local, dict) else None,
            "training_rows": data.get("training_rows"),
            "quality_note": data.get("quality_note"),
        }
    except Exception as exc:
        return {
            "ok": False,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }
