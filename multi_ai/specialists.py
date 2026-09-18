import os
from datetime import datetime, timezone
from typing import Any

import httpx


async def _get_json(url: str, timeout: float = 20.0) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.json()


async def _post_json(url: str, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload)
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
            "source": "AI_ML_V0_11_3",
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
            "source": "AI_ML_V0_11_3",
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
        "warning": "FinBERT/financial NLP endpoint not configured; baseline is not a production sentiment model.",
    }


async def nlp_sentiment(symbol: str, news: list[str]) -> dict[str, Any]:
    url = os.getenv("NLP_SERVICE_URL", "").strip()
    if not url:
        return _lexicon_sentiment(news)
    try:
        data = await _post_json(url, {"symbol": symbol, "texts": news})
        return {"status": "READY", "source": "NLP_SERVICE", **data}
    except Exception as exc:
        fallback = _lexicon_sentiment(news)
        fallback["service_error"] = f"{type(exc).__name__}: {exc}"
        return fallback


async def deep_forecast(symbol: str, candles: list[dict[str, Any]]) -> dict[str, Any]:
    url = os.getenv("DL_SERVICE_URL", "").strip()
    if not url:
        return {
            "status": "NOT_CONFIGURED",
            "source": "DEEP_FORECAST",
            "symbol": symbol,
            "model": None,
            "warning": "TFT/PatchTST/Chronos endpoint is not configured yet.",
        }
    try:
        data = await _post_json(url, {"symbol": symbol, "candles": candles})
        return {"status": "READY", "source": "DEEP_FORECAST", **data}
    except Exception as exc:
        return {
            "status": "ERROR",
            "source": "DEEP_FORECAST",
            "symbol": symbol,
            "error": f"{type(exc).__name__}: {exc}",
        }


def structure_context(extra: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "market_regime",
        "external_structure",
        "internal_structure",
        "mss",
        "bos",
        "choch",
        "order_block",
        "fvg",
        "liquidity_sweep",
        "session",
        "spread",
        "atr",
        "rr",
        "entry",
        "sl",
        "tp1",
        "tp2",
        "tp3",
    }
    return {k: extra.get(k) for k in allowed if k in extra}


async def build_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    symbol = str(payload.get("symbol", "XAUUSD")).upper()
    news = payload.get("news") if isinstance(payload.get("news"), list) else []
    news = [str(x)[:1200] for x in news[:30]]
    candles = payload.get("candles") if isinstance(payload.get("candles"), list) else []
    candles = candles[-500:]

    ml = await market_intelligence(symbol)
    nlp = await nlp_sentiment(symbol, news)
    dl = await deep_forecast(symbol, candles)

    return {
        "evidence_version": "1.0",
        "symbol": symbol,
        "timeframe": str(payload.get("timeframe", "M3")).upper(),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "machine_learning": ml,
        "nlp_sentiment": nlp,
        "deep_forecast": dl,
        "structure": structure_context(payload.get("structure") or {}),
        "operator_context": payload.get("context") if isinstance(payload.get("context"), dict) else {},
        "execution_requested": bool(payload.get("execution_requested", False)),
    }
