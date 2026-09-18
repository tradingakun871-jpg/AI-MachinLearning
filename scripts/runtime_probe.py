import asyncio
import json
import os
import time

import httpx


def base(name: str) -> str:
    return os.getenv(name, "").strip().rstrip("/")


def safe_summary(payload):
    if not isinstance(payload, dict):
        return {}
    return payload


async def get_json(client, url, timeout=20.0):
    t0 = time.perf_counter()
    r = await client.get(url, timeout=timeout)
    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    r.raise_for_status()
    return r.json(), elapsed, r.status_code


async def post_json(client, url, token, payload, timeout=90.0):
    t0 = time.perf_counter()
    r = await client.post(
        url,
        headers={"X-Internal-Token": token},
        json=payload,
        timeout=timeout,
    )
    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    r.raise_for_status()
    return r.json(), elapsed, r.status_code


async def main():
    await asyncio.sleep(float(os.getenv("RUNTIME_PROBE_DELAY", "8")))

    token = os.getenv("INTERNAL_SERVICE_TOKEN", "").strip()
    ml = base("MARKET_INTELLIGENCE_URL")
    nlp = base("NLP_SERVICE_URL")
    deep = base("DL_SERVICE_URL")

    result = {
        "probe": "multi-ai-runtime-v1",
        "ok": False,
        "ml": {},
        "nlp": {},
        "deep": {},
    }

    if not token or not ml or not nlp or not deep:
        result["error"] = "required runtime configuration missing"
        print("RUNTIME_PROBE " + json.dumps(result, separators=(",", ":")), flush=True)
        return

    try:
        async with httpx.AsyncClient() as client:
            # ML health/status
            try:
                data, latency, code = await get_json(client, f"{ml}/api/live/status", 20.0)
                result["ml"] = {
                    "ok": True,
                    "http": code,
                    "latency_ms": latency,
                    "status": data.get("status"),
                    "mode": data.get("mode"),
                    "version": data.get("version"),
                }
            except Exception as exc:
                result["ml"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}

            # FinBERT warmup + sample event-impact inference.
            try:
                _, warm_latency, warm_code = await post_json(client, f"{nlp}/warmup", token, {}, 120.0)
                data, latency, code = await post_json(
                    client,
                    f"{nlp}/analyze",
                    token,
                    {
                        "symbol": "XAUUSD",
                        "texts": ["US CPI comes in above forecast as inflation stays hot"],
                        "event": {"actual": 3.4, "forecast": 3.1},
                    },
                    60.0,
                )
                impact = data.get("event_impact") or {}
                result["nlp"] = {
                    "ok": True,
                    "warmup_http": warm_code,
                    "warmup_latency_ms": warm_latency,
                    "http": code,
                    "latency_ms": latency,
                    "model": data.get("model"),
                    "label": data.get("label"),
                    "score": data.get("score"),
                    "event_type": impact.get("event_type"),
                    "market_direction": impact.get("direction"),
                    "impact_strength": impact.get("impact_strength"),
                }
            except Exception as exc:
                result["nlp"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}

            # Deep warmup then a real XAUUSD M3 forecast using DB history on the specialist.
            try:
                _, warm_latency, warm_code = await post_json(client, f"{deep}/warmup", token, {}, 120.0)
                data, latency, code = await post_json(
                    client,
                    f"{deep}/forecast",
                    token,
                    {"symbol": "XAUUSD", "timeframe": "M3", "candles": [], "prediction_length": 10},
                    float(os.getenv("DEEP_FORECAST_TIMEOUT", "180")),
                )
                local = data.get("local_models") or data.get("local_training") or {}
                models = data.get("models") or {}
                result["deep"] = {
                    "ok": True,
                    "warmup_http": warm_code,
                    "warmup_latency_ms": warm_latency,
                    "http": code,
                    "latency_ms": latency,
                    "status": data.get("status"),
                    "direction": data.get("direction"),
                    "direction_confidence": data.get("direction_confidence"),
                    "ensemble_weights": data.get("ensemble_weights"),
                    "local_state": local.get("state") if isinstance(local, dict) else None,
                    "patchtst_present": "patchtst" in models,
                    "tft_present": "tft" in models,
                    "chronos_present": "chronos2" in models,
                    "training_rows": data.get("training_rows"),
                }
            except Exception as exc:
                result["deep"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}

        result["ok"] = all(result[k].get("ok") for k in ("ml", "nlp", "deep"))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:300]

    print("RUNTIME_PROBE " + json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
