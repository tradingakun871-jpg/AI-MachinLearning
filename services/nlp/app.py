import os
import re
import threading
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_ID = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
_pipe = None
_lock = threading.Lock()

app = FastAPI(title="Financial NLP Brain", version="1.0.0")


class NLPRequest(BaseModel):
    symbol: str = "XAUUSD"
    texts: list[str] = Field(default_factory=list, max_length=50)
    event: dict[str, Any] | None = None


def get_pipe():
    global _pipe
    if _pipe is None:
        with _lock:
            if _pipe is None:
                from transformers import pipeline
                _pipe = pipeline(
                    "text-classification",
                    model=MODEL_ID,
                    tokenizer=MODEL_ID,
                    top_k=None,
                    device=-1,
                )
    return _pipe


EVENTS = [
    ("US_INFLATION", [r"\bcpi\b", r"\bpce\b", r"inflation", r"consumer price"]),
    ("US_JOBS", [r"\bnfp\b", r"nonfarm", r"payroll", r"unemployment", r"jobless"]),
    ("FED_POLICY", [r"\bfomc\b", r"federal reserve", r"\bfed\b", r"powell", r"rate cut", r"rate hike"]),
    ("USD_YIELD", [r"\bdxy\b", r"dollar index", r"treasury", r"yield", r"us10y"]),
    ("GEO_RISK", [r"war", r"conflict", r"sanction", r"missile", r"ceasefire", r"geopolit"]),
    ("BTC_ETF", [r"bitcoin etf", r"btc etf", r"spot etf", r"etf inflow", r"etf outflow"]),
    ("CRYPTO_REGULATION", [r"sec ", r"regulation", r"regulator", r"ban", r"lawsuit", r"approval"]),
    ("CRYPTO_SECURITY", [r"hack", r"exploit", r"stolen", r"breach", r"exchange halt"]),
]


def event_type(text: str) -> str:
    low = text.lower()
    for name, pats in EVENTS:
        if any(re.search(p, low) for p in pats):
            return name
    return "GENERAL_FINANCIAL"


def label_scores(raw: list[dict[str, Any]]) -> dict[str, float]:
    out = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for item in raw:
        lab = str(item.get("label", "")).lower()
        if lab in out:
            out[lab] = float(item.get("score", 0.0))
    return out


def surprise(event: dict[str, Any] | None) -> float | None:
    if not event:
        return None
    try:
        actual = float(event.get("actual"))
        forecast = float(event.get("forecast"))
        scale = max(abs(forecast), 1e-9)
        return max(-3.0, min(3.0, (actual - forecast) / scale))
    except Exception:
        return None


def event_impact(symbol: str, kind: str, sent: float, evt: dict[str, Any] | None) -> dict[str, Any]:
    symbol = symbol.upper()
    s = surprise(evt)
    direction = "NEUTRAL"
    strength = min(1.0, abs(sent))
    rationale = "No strong asset-specific event mapping."

    if symbol == "XAUUSD":
        if kind in ("US_INFLATION", "US_JOBS", "FED_POLICY", "USD_YIELD"):
            macro = 0.0
            if s is not None:
                macro = s
            elif kind == "FED_POLICY":
                macro = -sent
            else:
                macro = -sent
            # Positive US macro / hawkish / rising yield usually pressures gold; inverse for weak/dovish.
            gold_bias = -macro
            if gold_bias > 0.08:
                direction = "BULLISH"
            elif gold_bias < -0.08:
                direction = "BEARISH"
            strength = min(1.0, max(abs(gold_bias), abs(sent)))
            rationale = "Mapped through USD/rates channel into gold sensitivity."
        elif kind == "GEO_RISK":
            direction = "BULLISH" if sent <= 0 else "NEUTRAL"
            strength = min(1.0, max(strength, 0.55 if sent < 0 else 0.25))
            rationale = "Risk-off/geopolitical stress can support safe-haven demand."
    elif symbol == "BTCUSD":
        if kind == "BTC_ETF":
            direction = "BULLISH" if sent > 0.05 else "BEARISH" if sent < -0.05 else "NEUTRAL"
            strength = min(1.0, max(strength, 0.6))
            rationale = "ETF flow/approval event mapped directly to BTC demand conditions."
        elif kind in ("CRYPTO_REGULATION", "CRYPTO_SECURITY"):
            direction = "BULLISH" if sent > 0.15 else "BEARISH" if sent < -0.05 else "NEUTRAL"
            strength = min(1.0, max(strength, 0.55))
            rationale = "Regulatory/security event mapped to crypto-specific risk premium."
        elif kind in ("FED_POLICY", "USD_YIELD", "US_INFLATION", "US_JOBS"):
            macro = s if s is not None else -sent
            btc_bias = -macro
            direction = "BULLISH" if btc_bias > 0.08 else "BEARISH" if btc_bias < -0.08 else "NEUTRAL"
            strength = min(1.0, max(abs(btc_bias), abs(sent)))
            rationale = "Macro liquidity/rates event mapped into BTC risk-asset sensitivity."

    return {
        "event_type": kind,
        "direction": direction,
        "impact_strength": round(float(strength), 4),
        "surprise": None if s is None else round(float(s), 4),
        "rationale": rationale,
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "financial-nlp-brain", "model": MODEL_ID, "model_loaded": _pipe is not None}


@app.post("/warmup")
def warmup():
    get_pipe()
    return {"ok": True, "model": MODEL_ID, "loaded": True}


@app.post("/analyze")
def analyze(req: NLPRequest):
    texts = [x.strip() for x in req.texts if x and x.strip()][:50]
    if not texts:
        return {
            "symbol": req.symbol.upper(),
            "status": "NO_DATA",
            "model": MODEL_ID,
            "score": 0.0,
            "label": "NEUTRAL",
            "items": [],
            "event_impact": event_impact(req.symbol, "GENERAL_FINANCIAL", 0.0, req.event),
        }
    try:
        raw = get_pipe()(texts, truncation=True, max_length=512)
    except Exception as exc:
        raise HTTPException(503, f"FinBERT inference failed: {type(exc).__name__}: {exc}")

    items = []
    agg = 0.0
    weight = 0.0
    for text, scores_raw in zip(texts, raw):
        scores = label_scores(scores_raw)
        signed = scores["positive"] - scores["negative"]
        conf = max(scores.values())
        kind = event_type(text)
        impact = event_impact(req.symbol, kind, signed, req.event)
        items.append({
            "text": text,
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "signed_score": round(signed, 4),
            "event": impact,
        })
        w = 0.25 + conf * 0.75
        agg += signed * w
        weight += w

    score = agg / weight if weight else 0.0
    label = "POSITIVE" if score > 0.12 else "NEGATIVE" if score < -0.12 else "NEUTRAL"
    dominant = max(items, key=lambda x: x["event"]["impact_strength"])["event"] if items else None
    return {
        "status": "READY",
        "symbol": req.symbol.upper(),
        "model": MODEL_ID,
        "score": round(score, 4),
        "label": label,
        "items": items,
        "event_impact": dominant,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
