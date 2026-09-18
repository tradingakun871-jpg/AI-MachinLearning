import asyncio
import json
import os
import re
from typing import Any

import httpx

VALID_DECISIONS = {"BUY", "SELL", "NO_TRADE"}


def _json_from_text(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model response")
    try:
        return json.loads(text)
    except Exception:
        pass
    fenced = re.search(r"\`\`\`(?:json)?\s*(\{.*?\})\s*\`\`\`", text, re.S | re.I)
    if fenced:
        return json.loads(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("model did not return valid JSON")


def normalize_vote(provider: str, payload: dict[str, Any]) -> dict[str, Any]:
    decision = str(payload.get("decision", "NO_TRADE")).upper().replace(" ", "_")
    if decision == "WAIT":
        decision = "NO_TRADE"
    if decision not in VALID_DECISIONS:
        decision = "NO_TRADE"
    try:
        confidence = float(payload.get("confidence", 0))
    except Exception:
        confidence = 0.0
    if 0 <= confidence <= 1:
        confidence *= 100
    confidence = max(0.0, min(100.0, confidence))
    checks = payload.get("checks") if isinstance(payload.get("checks"), dict) else {}
    return {
        "provider": provider,
        "decision": decision,
        "confidence": round(confidence, 2),
        "veto": bool(payload.get("veto", False)),
        "reason": str(payload.get("reason", "")).strip()[:1200],
        "checks": checks,
        "raw_valid": True,
    }


async def _post(url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout: float = 45.0):
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, headers=headers, json=json_body)
        r.raise_for_status()
        return r.json()


async def call_openai(prompt: str) -> dict[str, Any]:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "").strip()
    if not key or not model:
        raise RuntimeError("OPENAI_API_KEY/OPENAI_MODEL not configured")
    data = await _post(
        os.getenv("OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json_body={"model": model, "input": prompt},
    )
    text = data.get("output_text")
    if not text:
        chunks = []
        for item in data.get("output", []) or []:
            for c in item.get("content", []) or []:
                if c.get("type") in ("output_text", "text") and c.get("text"):
                    chunks.append(c["text"])
        text = "\n".join(chunks)
    return normalize_vote("chatgpt", _json_from_text(text or ""))


async def call_claude(prompt: str) -> dict[str, Any]:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    model = os.getenv("ANTHROPIC_MODEL", "").strip()
    if not key or not model:
        raise RuntimeError("ANTHROPIC_API_KEY/ANTHROPIC_MODEL not configured")
    data = await _post(
        os.getenv("ANTHROPIC_MESSAGES_URL", "https://api.anthropic.com/v1/messages"),
        headers={
            "x-api-key": key,
            "anthropic-version": os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
            "Content-Type": "application/json",
        },
        json_body={
            "model": model,
            "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS", "900")),
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    text = "\n".join(x.get("text", "") for x in data.get("content", []) if x.get("type") == "text")
    return normalize_vote("claude", _json_from_text(text))


async def call_gemini(prompt: str) -> dict[str, Any]:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "").strip()
    if not key or not model:
        raise RuntimeError("GEMINI_API_KEY/GEMINI_MODEL not configured")
    base = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    url = f"{base}/models/{model}:generateContent?key={key}"
    data = await _post(
        url,
        headers={"Content-Type": "application/json"},
        json_body={
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        },
    )
    candidates = data.get("candidates", []) or []
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
    text = "\n".join(p.get("text", "") for p in parts if p.get("text"))
    return normalize_vote("gemini", _json_from_text(text))


async def call_qwen(prompt: str) -> dict[str, Any]:
    key = os.getenv("QWEN_API_KEY", "").strip()
    model = os.getenv("QWEN_MODEL", "").strip()
    base = os.getenv("QWEN_BASE_URL", "").strip().rstrip("/")
    if not key or not model or not base:
        raise RuntimeError("QWEN_API_KEY/QWEN_MODEL/QWEN_BASE_URL not configured")
    data = await _post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json_body={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        },
    )
    text = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    return normalize_vote("qwen3", _json_from_text(text))


CALLERS = {
    "chatgpt": call_openai,
    "claude": call_claude,
    "gemini": call_gemini,
    "qwen3": call_qwen,
}


async def ask_all(prompt: str) -> list[dict[str, Any]]:
    async def one(name: str, fn):
        try:
            return await fn(prompt)
        except Exception as exc:
            return {
                "provider": name,
                "decision": "NO_TRADE",
                "confidence": 0.0,
                "veto": True,
                "reason": f"PROVIDER_ERROR: {type(exc).__name__}: {exc}",
                "checks": {},
                "raw_valid": False,
            }

    return await asyncio.gather(*(one(name, fn) for name, fn in CALLERS.items()))
