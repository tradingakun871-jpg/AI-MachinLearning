import io
import os
import threading
from typing import Any

import psycopg
import torch

import compat_app as compat

core = compat.core
app = compat.app
_db_lock = threading.Lock()
_db_ready = False
_original_get_cached_bundle = core.get_cached_bundle

DDL = """
CREATE TABLE IF NOT EXISTS deep_model_artifacts(
    artifact_key TEXT PRIMARY KEY,
    payload BYTEA NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


def _db_url() -> str:
    return os.getenv("DATABASE_URL", "").strip()


def _ensure_db() -> bool:
    global _db_ready
    if _db_ready:
        return True
    url = _db_url()
    if not url:
        return False
    with _db_lock:
        if _db_ready:
            return True
        with psycopg.connect(url) as conn:
            with conn.cursor() as cur:
                cur.execute(DDL)
            conn.commit()
        _db_ready = True
    return True


def _serialize_bundle(bundle: dict[str, Any], horizon: int) -> bytes:
    payload = {
        "format_version": 1,
        "horizon": int(horizon),
        "context": int(bundle["context"]),
        "ret_scale": float(bundle["ret_scale"]),
        "losses": bundle.get("losses") or {},
        "trained_rows": int(bundle.get("trained_rows") or 0),
        "trained_last_ts": bundle.get("trained_last_ts"),
        "trained_at": bundle.get("trained_at"),
        "patchtst_state": bundle["patchtst"].state_dict(),
        "tft_state": bundle["tft"].state_dict(),
    }
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


def _deserialize_bundle(raw: bytes) -> dict[str, Any]:
    buf = io.BytesIO(raw)
    try:
        payload = torch.load(buf, map_location=core.DEVICE, weights_only=True)
    except TypeError:
        buf.seek(0)
        payload = torch.load(buf, map_location=core.DEVICE)
    horizon = int(payload["horizon"])
    context = int(payload["context"])
    patch = core.PatchTSTMini(context, horizon).to(core.DEVICE)
    tft = core.TemporalFusionTransformerMini(4, horizon).to(core.DEVICE)
    patch.load_state_dict(payload["patchtst_state"])
    tft.load_state_dict(payload["tft_state"])
    patch.eval()
    tft.eval()
    return {
        "patchtst": patch,
        "tft": tft,
        "ret_scale": float(payload["ret_scale"]),
        "context": context,
        "losses": payload.get("losses") or {},
        "trained_rows": int(payload.get("trained_rows") or 0),
        "trained_last_ts": payload.get("trained_last_ts"),
        "trained_at": payload.get("trained_at"),
        "artifact_source": "POSTGRES",
    }


def _save_db(key: str, bundle: dict[str, Any], horizon: int) -> bool:
    if not _ensure_db():
        return False
    raw = _serialize_bundle(bundle, horizon)
    sql = """
    INSERT INTO deep_model_artifacts(artifact_key,payload,updated_at)
    VALUES(%s,%s,NOW())
    ON CONFLICT(artifact_key) DO UPDATE SET payload=EXCLUDED.payload, updated_at=NOW()
    """
    with psycopg.connect(_db_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (key, raw))
        conn.commit()
    return True


def _load_db(key: str) -> dict[str, Any] | None:
    if not _ensure_db():
        return None
    with psycopg.connect(_db_url()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT payload FROM deep_model_artifacts WHERE artifact_key=%s", (key,))
            row = cur.fetchone()
    if not row:
        return None
    try:
        return _deserialize_bundle(bytes(row[0]))
    except Exception as exc:
        print(f"DEEP_DB_ARTIFACT_LOAD_ERROR key={key} error={type(exc).__name__}:{exc}", flush=True)
        return None


def db_get_cached_bundle(symbol: str, timeframe: str, df, horizon: int):
    bundle, freshness = _original_get_cached_bundle(symbol, timeframe, df, horizon)
    if bundle is not None:
        return bundle, freshness
    key = core._cache_key(symbol, timeframe, horizon)
    bundle = _load_db(key)
    if bundle is None:
        return None, "MISSING"
    with core._cache_lock:
        core._cache[key] = {"bundle": bundle, "status": "READY", "source": "POSTGRES"}
    new_bars = core._bars_since(df, bundle.get("trained_last_ts"))
    return bundle, "FRESH" if new_bars < core.LOCAL_RETRAIN_BARS else "STALE"


def db_train_worker(key: str, df, horizon: int):
    try:
        bundle = core.fit_local(df, horizon)
        persisted = _save_db(key, bundle, horizon)
        with core._cache_lock:
            core._cache[key] = {
                "bundle": bundle,
                "status": "READY",
                "source": "TRAINED+POSTGRES" if persisted else "TRAINED_RAM_ONLY",
            }
        print(f"DEEP_TRAIN_READY key={key} persisted={persisted}", flush=True)
    except Exception as exc:
        with core._cache_lock:
            core._cache[key] = {"bundle": None, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
        print(f"DEEP_TRAIN_ERROR key={key} error={type(exc).__name__}:{exc}", flush=True)
    finally:
        with core._cache_lock:
            core._training.discard(key)


core.get_cached_bundle = db_get_cached_bundle
core._train_worker = db_train_worker


@app.get("/artifact-status")
def artifact_status():
    try:
        if not _ensure_db():
            return {"ok": False, "backend": "POSTGRES", "error": "DATABASE_URL unavailable"}
        with psycopg.connect(_db_url()) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT artifact_key, octet_length(payload), updated_at FROM deep_model_artifacts ORDER BY updated_at DESC LIMIT 20")
                rows = cur.fetchall()
        return {
            "ok": True,
            "backend": "POSTGRES",
            "count": len(rows),
            "artifacts": [
                {"key": r[0], "bytes": int(r[1]), "updated_at": r[2].isoformat()} for r in rows
            ],
        }
    except Exception as exc:
        return {"ok": False, "backend": "POSTGRES", "error": f"{type(exc).__name__}: {exc}"[:500]}
