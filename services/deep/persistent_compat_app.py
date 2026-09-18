import os
import re
from pathlib import Path
from typing import Any

import torch

import compat_app as compat

core = compat.core
ARTIFACT_DIR = Path(os.getenv("DEEP_ARTIFACT_DIR", "/app/artifacts/deep"))
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

_original_get_cached_bundle = core.get_cached_bundle


def _safe_key(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key)


def _artifact_path(key: str) -> Path:
    return ARTIFACT_DIR / f"{_safe_key(key)}.pt"


def _save_bundle(key: str, bundle: dict[str, Any], horizon: int) -> None:
    path = _artifact_path(key)
    tmp = path.with_suffix(".tmp")
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
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _load_bundle(key: str) -> dict[str, Any] | None:
    path = _artifact_path(key)
    if not path.exists():
        return None
    try:
        try:
            payload = torch.load(path, map_location=core.DEVICE, weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location=core.DEVICE)
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
            "artifact_path": str(path),
        }
    except Exception as exc:
        print(f"DEEP_ARTIFACT_LOAD_ERROR key={key} error={type(exc).__name__}:{exc}", flush=True)
        return None


def persistent_get_cached_bundle(symbol: str, timeframe: str, df, horizon: int):
    bundle, freshness = _original_get_cached_bundle(symbol, timeframe, df, horizon)
    if bundle is not None:
        return bundle, freshness
    key = core._cache_key(symbol, timeframe, horizon)
    bundle = _load_bundle(key)
    if bundle is None:
        return None, "MISSING"
    with core._cache_lock:
        core._cache[key] = {"bundle": bundle, "status": "READY", "source": "ARTIFACT"}
    new_bars = core._bars_since(df, bundle.get("trained_last_ts"))
    return bundle, "FRESH" if new_bars < core.LOCAL_RETRAIN_BARS else "STALE"


def persistent_train_worker(key: str, df, horizon: int):
    try:
        bundle = core.fit_local(df, horizon)
        _save_bundle(key, bundle, horizon)
        with core._cache_lock:
            core._cache[key] = {"bundle": bundle, "status": "READY", "source": "TRAINED"}
        print(f"DEEP_ARTIFACT_SAVED key={key} path={_artifact_path(key)}", flush=True)
    except Exception as exc:
        with core._cache_lock:
            core._cache[key] = {"bundle": None, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
        print(f"DEEP_TRAIN_ERROR key={key} error={type(exc).__name__}:{exc}", flush=True)
    finally:
        with core._cache_lock:
            core._training.discard(key)


core.get_cached_bundle = persistent_get_cached_bundle
core._train_worker = persistent_train_worker

app = compat.app


@app.get("/artifact-status")
def artifact_status():
    files = sorted(p.name for p in ARTIFACT_DIR.glob("*.pt"))
    return {
        "artifact_dir": str(ARTIFACT_DIR),
        "persistent_files": files,
        "count": len(files),
    }
