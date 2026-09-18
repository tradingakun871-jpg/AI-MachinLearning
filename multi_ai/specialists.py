import asyncio
import os
from datetime import datetime, timezone
from typing import Any

import httpx


def _base_url(name: str) -> str:
    return os.getenv(name, "").strip().rstrip("/")


async def _get_json(url: str, timeout: float = 20.0) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.json()


async def _post_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 90.0) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload or {})
        r.raise_for_status()
        return r.json()


async def market_intelligence(symbol: str) -> dict[str, Any]:
    base = os.getenv(
        "MARKET_INTELLIGENCE_URL",
        "https://ai-machine-learning-production.up.railway.app",
    ).rstrip("/")
    try:
        status = await _get_json(f"{base}/api/live/status")
        latest = status.get("last_signal")
        training = (status.get("training") or {}).get(symbol) or {}
        model = (status.get("models") or {}).get(symbol) or {}
        return {
            "status": "READY",
            "source": "AI_ML",
            "service_version": status.get("version"),
            "symbol": symbol,
            "model": model,
            "training": {
                "status": training.get("status"),
                "version": training.get("version"),
                "quality": training.get("quality"),
            },
            "latest_signal": latest if isinstance(latest, dict) and str(latest.get("symbol", "")).upper() == symbol else None,
            "market_sources": status.get("sources"),
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        return {
            "status": "ERROR",
            "source": "AI_ML",
            "symbol": symbol,
            "error": f"{type(exc).__name__}: {exc}",
        }


POSITIVE = {
    "beat", "beats", "strong", "growth", "bullish", "surge", "rally", "inflow",
    "easing", "cut", "dovish", "approval", "approved", "record high", "demand",
}
NEGATIVE = {
    "miss", "weak", "bearish", "drop", "selloff", "outflow", "hawkish",
    "hike", "inflation", "ban", "hack", "liquidation", "recession", "default",
}


def _lexicon_sentiment(news: list[str]) -> dict[str, Any]:
    if not news:
        return {
            "status": "NO_DATA",
            "model": "LEXICON_BASELINE",
            "score": 0.0,
            "label": "NEUTRAL",
            "items": 0,
        }
    total = 0.0
    for item in news:
        t = item.lower()
        pos = sum(1 for w in POSITIVE if w in t)
        neg = sum(1 for w in NEGATIVE if w in t)
        total += pos - neg
    score = max(-1.0, min(1.0, total / max(1, len(news) * 2)))
    label = "POSITIVE" if score > 0.15 else "NEGATIVE" if score < -0.15 else "NEUTRAL"
    return {
        "status": "BASELINE_ONLY",
        "model": "LEXICON_BASELINE",
        "score": round(score, 4),
        "label": label,
        "items": len(news),
        "warning": "Financial NLP service unavailable; lexicon fallback is research-only.",
    }


async def nlp_sentiment(symbol: str, news: list[str], event: dict[str, Any] | None = None) -> dict[str, Any]:
    base = _base_url("NLP_SERVICE_URL")
    if not base:
        return _lexicon_sentiment(news)
    try:
        data = await _post_json(
            f"{base}/analyze",
            {"symbol": symbol, "texts": news, "event": event},
            timeout=120.0,
        )
        return {"source": "FINANCIAL_NLP_BRAIN", **data}
    except Exception as exc:
        fallback = _lexicon_sentiment(news)
        fallback["service_error"] = f"{type(exc).__name__}: {exc}"
        return fallback


async def deep_forecast(symbol: str, timeframe: str, candles: list[dict[str, Any]]) -> dict[str, Any]:
    base = _base_url("DL_SERVICE_URL")
    if not base:
        return {
            "status": "NOT_CONFIGURED",
            "source": "DEEP_FORECAST",
            "symbol": symbol,
            "model": None,
            "warning": "Deep forecast endpoint is not configured.",
        }
    try:
        data = await _post_json(
            f"{base}/forecast",
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": candles,
                "prediction_length": int(os.getenv("DEEP_PREDICTION_LENGTH", "10")),
            },
            timeout=float(os.getenv("DEEP_FORECAST_TIMEOUT", "180")),
        )
        return {"source": "DEEP_FORECAST_BRAIN", **data}
    except Exception as exc:
        return {
            "status": "ERROR",
            "source": "DEEP_FORECAST_BRAIN",
            "symbol": symbol,
            "timeframe": timeframe,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def specialist_health() -> dict[str, Any]:
    services = {
        "machine_learning": os.getenv("MARKET_INTELLIGENCE_URL", "").strip().rstrip("/"),
        "financial_nlp": _base_url("NLP_SERVICE_URL"),
        "deep_forecast": _base_url("DL_SERVICE_URL"),
    }

    async def check(name: str, base: str):
        if not base:
            return name, {"configured": False, "reachable": False, "status": "NOT_CONFIGURED"}
        path = "/health" if name != "machine_learning" else "/health"
        try:
            data = await _get_json(f"{base}{path}", timeout=12.0)
            return name, {"configured": True, "reachable": True, "status": "ONLINE", "details": data}
        except Exception as exc:
            return name, {"configured": True, "reachable": False, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}

    pairs = await asyncio.gather(*(check(name, base) for name, base in services.items()))
    return dict(pairs)


async def warmup_specialists() -> dict[str, Any]:
    targets = {
        "financial_nlp": _base_url("NLP_SERVICE_URL"),
        "deep_forecast": _base_url("DL_SERVICE_URL"),
    }

    async def warm(name: str, base: str):
        if not base:
            return name, {"ok": False, "error": "NOT_CONFIGURED"}
        try:
            data = await _post_json(f"{base}/warmup", {}, timeout=300.0)
            return name, {"ok": True, "details": data}
        except Exception as exc:
            return name, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    pairs = await asyncio.gather(*(warm(name, base) for name, base in targets.items()))
    result = dict(pairs)
    result["ready"] = all(v.get("ok") for v in result.values() if isinstance(v, dict))
    return result


def structure_context(extra: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "market_regime", "external_structure", "internal_structure", "mss", "bos", "choch",
        "order_block", "fvg", "liquidity_sweep", "session", "spread", "atr", "rr",
        "entry", "sl", "tp1", "tp2", "tp3",
    }
    return {k: extra.get(k) for k in allowed if k in extra}


async def build_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    symbol = str(payload.get("symbol", "XAUUSD")).upper()
    timeframe = str(payload.get("timeframe", "M3")).upper()
    news = payload.get("news") if isinstance(payload.get("news"), list) else []
    news = [str(x)[:1200] for x in news[:30]]
    candles = payload.get("candles") if isinstance(payload.get("candles"), list) else []
    candles = candles[-500:]
    event = payload.get("event") if isinstance(payload.get("event"), dict) else None

    # Same captured request context; independent specialists run concurrently to reduce latency.
    ml, nlp, dl = await asyncio.gather(
        market_intelligence(symbol),
        nlp_sentiment(symbol, news, event),
        deep_forecast(symbol, timeframe, candles),
    )

    return {
        "evidence_version": "1.1",
        "symbol": symbol,
        "timeframe": timeframe,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "machine_learning": ml,
        "nlp_sentiment": nlp,
        "deep_forecast": dl,
        "structure": structure_context(payload.get("structure") or {}),
        "operator_context": payload.get("context") if isinstance(payload.get("context"), dict) else {},
        "execution_requested": bool(payload.get("execution_requested", False)),
    }
