import asyncio
import json
import os
import time

import httpx


async def main():
    ml = os.getenv("MARKET_INTELLIGENCE_URL", "").rstrip("/")
    nlp = os.getenv("NLP_SERVICE_URL", "").rstrip("/")
    deep = os.getenv("DL_SERVICE_URL", "").rstrip("/")
    token = os.getenv("INTERNAL_SERVICE_TOKEN", "").strip()
    if not all([ml, nlp, deep, token]):
        raise RuntimeError("specialist URLs or INTERNAL_SERVICE_TOKEN missing")

    headers = {"X-Internal-Token": token, "Content-Type": "application/json"}
    out = {"ok": False, "checks": {}}
    timeout = httpx.Timeout(90.0, connect=15.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        t = time.perf_counter()
        r = await client.get(f"{ml}/health")
        r.raise_for_status()
        out["checks"]["ml"] = {"status": r.status_code, "latency_ms": round((time.perf_counter()-t)*1000, 1)}

        t = time.perf_counter()
        r = await client.post(f"{nlp}/warmup", headers=headers, json={})
        r.raise_for_status()
        out["checks"]["nlp_warmup"] = {"status": r.status_code, "latency_ms": round((time.perf_counter()-t)*1000, 1)}

        t = time.perf_counter()
        r = await client.post(
            f"{nlp}/analyze",
            headers=headers,
            json={
                "symbol": "XAUUSD",
                "texts": ["US CPI comes in above forecast as inflation stays hot"],
                "event": {"actual": 3.4, "forecast": 3.1},
            },
        )
        r.raise_for_status()
        data = r.json()
        out["checks"]["nlp_analyze"] = {
            "status": r.status_code,
            "latency_ms": round((time.perf_counter()-t)*1000, 1),
            "model_status": data.get("status"),
            "event_direction": (data.get("event_impact") or {}).get("direction"),
        }

        t = time.perf_counter()
        r = await client.post(f"{deep}/warmup", headers=headers, json={})
        r.raise_for_status()
        out["checks"]["deep_warmup"] = {"status": r.status_code, "latency_ms": round((time.perf_counter()-t)*1000, 1)}

        t = time.perf_counter()
        r = await client.post(
            f"{deep}/forecast",
            headers=headers,
            json={"symbol": "XAUUSD", "timeframe": "M3", "candles": [], "prediction_length": 5},
        )
        r.raise_for_status()
        data = r.json()
        latency = time.perf_counter() - t
        local = data.get("local_model_status") or {}
        out["checks"]["deep_forecast"] = {
            "status": r.status_code,
            "latency_ms": round(latency*1000, 1),
            "model_status": data.get("status"),
            "direction": data.get("direction"),
            "local_state": local.get("state"),
            "ensemble_weights": data.get("ensemble_weights"),
        }
        if latency > 60:
            raise RuntimeError(f"deep forecast too slow: {latency:.1f}s")

    out["ok"] = True
    print("SPECIALIST_SMOKE=" + json.dumps(out, separators=(",", ":"), default=str))


if __name__ == "__main__":
    asyncio.run(main())
