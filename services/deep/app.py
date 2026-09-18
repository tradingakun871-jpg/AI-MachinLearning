import hmac
import math
import os
import threading
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

CHRONOS_MODEL = os.getenv("CHRONOS_MODEL", "autogluon/chronos-2-small")
LOCAL_EPOCHS = int(os.getenv("LOCAL_DEEP_EPOCHS", "2"))
MAX_WINDOWS = int(os.getenv("LOCAL_DEEP_MAX_WINDOWS", "96"))
LOCAL_RETRAIN_BARS = int(os.getenv("LOCAL_DEEP_RETRAIN_BARS", "20"))
DEVICE = "cpu"

app = FastAPI(title="Deep Forecast Brain", version="1.1.0")
_chronos = None
_chronos_lock = threading.Lock()
_cache: dict[str, dict[str, Any]] = {}
_cache_lock = threading.Lock()
_training: set[str] = set()


class ForecastRequest(BaseModel):
    symbol: str = "XAUUSD"
    timeframe: str = "M3"
    candles: list[dict[str, Any]] = Field(default_factory=list, max_length=2000)
    prediction_length: int = Field(default=10, ge=3, le=30)


def require_internal(x_internal_token: str | None = Header(default=None)):
    expected = os.getenv("INTERNAL_SERVICE_TOKEN", "").strip()
    if not expected:
        raise HTTPException(503, "internal service token is not configured")
    supplied = (x_internal_token or "").strip()
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(401, "unauthorized")


def load_db(symbol: str, timeframe: str, limit: int = 1200) -> list[dict[str, Any]]:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        return []
    import psycopg
    sql = """
    SELECT timestamp,open,high,low,close,volume,spread
    FROM market_candles
    WHERE symbol=%s AND timeframe=%s
    ORDER BY timestamp DESC LIMIT %s
    """
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (symbol.upper(), timeframe.upper(), limit))
            rows = cur.fetchall()
    rows.reverse()
    return [
        {"timestamp": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5], "spread": r[6]}
        for r in rows
    ]


def normalize_candles(candles: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for c in candles:
        try:
            ts = pd.to_datetime(c.get("timestamp") or c.get("time"), utc=True)
            close = float(c["close"])
            high = float(c.get("high", close))
            low = float(c.get("low", close))
            vol = float(c.get("volume", 0) or 0)
            rows.append((ts, close, high, low, vol))
        except Exception:
            continue
    if len(rows) < 80:
        raise ValueError("minimum 80 valid candles required")
    df = pd.DataFrame(rows, columns=["timestamp", "close", "high", "low", "volume"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    if len(df) < 80:
        raise ValueError("minimum 80 unique candles required")
    ret = np.log(df["close"].clip(lower=1e-12)).diff().fillna(0.0)
    vol = ret.rolling(20, min_periods=2).std().fillna(ret.std() or 1e-6).clip(lower=1e-7)
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["ret"] = ret
    df["rvol"] = vol
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    return df


class PatchTSTMini(nn.Module):
    def __init__(self, context: int, horizon: int, patch: int = 16, stride: int = 8, d_model: int = 48):
        super().__init__()
        self.context, self.horizon, self.patch, self.stride = context, horizon, patch, stride
        self.proj = nn.Linear(patch, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=96, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        p = x.unfold(1, self.patch, self.stride)
        z = self.proj(p)
        z = self.encoder(z)
        z = self.norm(z[:, -1])
        return self.head(z)


class TemporalFusionTransformerMini(nn.Module):
    def __init__(self, n_features: int, horizon: int, hidden: int = 32):
        super().__init__()
        self.horizon = horizon
        self.embeds = nn.ModuleList([nn.Linear(1, hidden) for _ in range(n_features)])
        self.selector = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(), nn.Linear(hidden, n_features))
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, dropout=0.1, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        weights = torch.softmax(self.selector(x), dim=-1)
        parts = [self.embeds[i](x[..., i:i+1]) * weights[..., i:i+1] for i in range(x.shape[-1])]
        z = torch.stack(parts, dim=0).sum(dim=0)
        h, _ = self.lstm(z)
        q = h[:, -1:, :]
        a, _ = self.attn(q, h, h, need_weights=False)
        cat = torch.cat([q, a], dim=-1)
        g = self.gate(cat)
        fused = self.norm(g * a + (1.0 - g) * q).squeeze(1)
        return self.head(fused)


def make_windows(df: pd.DataFrame, context: int, horizon: int):
    features = df[["ret", "rvol", "hour_sin", "hour_cos"]].to_numpy(np.float32)
    target = df["ret"].to_numpy(np.float32)
    xs, ys = [], []
    start = max(0, len(df) - (MAX_WINDOWS + context + horizon))
    for end in range(start + context, len(df) - horizon + 1):
        xs.append(features[end-context:end])
        ys.append(target[end:end+horizon])
    if not xs:
        raise ValueError("not enough history for supervised windows")
    return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys))


def fit_local(df: pd.DataFrame, horizon: int):
    torch.manual_seed(42)
    context = min(96, max(48, len(df) // 4))
    x, y = make_windows(df, context, horizon)
    ret_scale = float(df["ret"].std()) or 1e-5
    x = x.clone()
    x[..., 0] /= ret_scale
    x[..., 1] /= ret_scale
    y_scaled = y / ret_scale

    patch = PatchTSTMini(context, horizon).to(DEVICE)
    tft = TemporalFusionTransformerMini(4, horizon).to(DEVICE)
    models = [("patchtst", patch, lambda a: a[..., 0]), ("tft", tft, lambda a: a)]
    losses = {}
    batch = min(48, len(x))
    for name, model, select_x in models:
        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
        model.train()
        last_loss = None
        for _ in range(max(1, LOCAL_EPOCHS)):
            perm = torch.randperm(len(x))
            epoch_loss = 0.0
            steps = 0
            for j in range(0, len(x), batch):
                idx = perm[j:j+batch]
                xb = select_x(x[idx]).to(DEVICE)
                yb = y_scaled[idx].to(DEVICE)
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = torch.mean((pred - yb) ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                epoch_loss += float(loss.detach())
                steps += 1
            last_loss = epoch_loss / max(1, steps)
        model.eval()
        losses[name] = round(float(last_loss or 0.0), 6)

    return {
        "patchtst": patch,
        "tft": tft,
        "ret_scale": ret_scale,
        "context": context,
        "losses": losses,
        "trained_rows": len(df),
        "trained_last_ts": str(df["timestamp"].iloc[-1]),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }


def local_predict(bundle: dict[str, Any], df: pd.DataFrame, horizon: int):
    context = int(bundle["context"])
    tail = df[["ret", "rvol", "hour_sin", "hour_cos"]].to_numpy(np.float32)[-context:]
    x = torch.tensor(tail[None, ...])
    scale = float(bundle["ret_scale"])
    x[..., 0] /= scale
    x[..., 1] /= scale
    with torch.no_grad():
        patch = bundle["patchtst"](x[..., 0]).cpu().numpy().reshape(-1)[:horizon] * scale
        tft = bundle["tft"](x).cpu().numpy().reshape(-1)[:horizon] * scale
    return patch, tft


def _cache_key(symbol: str, timeframe: str, horizon: int) -> str:
    return f"{symbol}:{timeframe}:{horizon}"


def _bars_since(df: pd.DataFrame, trained_last_ts: str | None) -> int:
    if not trained_last_ts:
        return 10**9
    try:
        ts = pd.to_datetime(trained_last_ts, utc=True)
        return int((df["timestamp"] > ts).sum())
    except Exception:
        return 10**9


def get_cached_bundle(symbol: str, timeframe: str, df: pd.DataFrame, horizon: int):
    key = _cache_key(symbol, timeframe, horizon)
    with _cache_lock:
        item = _cache.get(key)
        bundle = item.get("bundle") if item else None
    if not bundle:
        return None, "MISSING"
    new_bars = _bars_since(df, bundle.get("trained_last_ts"))
    return bundle, "FRESH" if new_bars < LOCAL_RETRAIN_BARS else "STALE"


def _train_worker(key: str, df: pd.DataFrame, horizon: int):
    try:
        bundle = fit_local(df, horizon)
        with _cache_lock:
            _cache[key] = {"bundle": bundle, "status": "READY"}
    except Exception as exc:
        with _cache_lock:
            _cache[key] = {"bundle": None, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        with _cache_lock:
            _training.discard(key)


def schedule_local_training(symbol: str, timeframe: str, df: pd.DataFrame, horizon: int) -> bool:
    key = _cache_key(symbol, timeframe, horizon)
    with _cache_lock:
        if key in _training:
            return False
        _training.add(key)
    thread = threading.Thread(target=_train_worker, args=(key, df.copy(), horizon), daemon=True, name=f"local-train-{key}")
    thread.start()
    return True


def local_training_status(symbol: str, timeframe: str, horizon: int) -> dict[str, Any]:
    key = _cache_key(symbol, timeframe, horizon)
    with _cache_lock:
        training = key in _training
        item = _cache.get(key) or {}
    bundle = item.get("bundle")
    return {
        "state": "TRAINING" if training else item.get("status", "NOT_TRAINED"),
        "trained_rows": bundle.get("trained_rows") if bundle else None,
        "trained_at": bundle.get("trained_at") if bundle else None,
        "losses": bundle.get("losses") if bundle else None,
    }


def get_chronos():
    global _chronos
    if _chronos is None:
        with _chronos_lock:
            if _chronos is None:
                from chronos import Chronos2Pipeline
                _chronos = Chronos2Pipeline.from_pretrained(CHRONOS_MODEL, device_map="cpu")
    return _chronos


def chronos_predict(df: pd.DataFrame, horizon: int):
    pipe = get_chronos()
    context = pd.DataFrame({
        "id": ["series"] * len(df),
        "timestamp": df["timestamp"],
        "target": df["ret"].astype(float),
    })
    pred = pipe.predict_df(
        context,
        prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )
    med_col = "predictions" if "predictions" in pred.columns else "0.5"
    med = pred[med_col].to_numpy(dtype=float)[-horizon:]
    q10 = pred["0.1"].to_numpy(dtype=float)[-horizon:] if "0.1" in pred.columns else med
    q90 = pred["0.9"].to_numpy(dtype=float)[-horizon:] if "0.9" in pred.columns else med
    return med, q10, q90


def prices_from_returns(last_price: float, rets: np.ndarray):
    return (last_price * np.exp(np.cumsum(np.asarray(rets, dtype=float)))).tolist()


def direction_info(paths: list[np.ndarray], ensemble: np.ndarray, hist_vol: float):
    finals = np.array([float(np.sum(p)) for p in paths], dtype=float)
    agreement_up = float(np.mean(finals > 0))
    total = float(np.sum(ensemble))
    z = total / max(hist_vol * math.sqrt(max(1, len(ensemble))), 1e-8)
    strength = 1.0 / (1.0 + math.exp(-abs(z)))
    direction = "UP" if total > 0 else "DOWN" if total < 0 else "FLAT"
    confidence = (0.55 * max(agreement_up, 1.0 - agreement_up) + 0.45 * strength) * 100.0
    return direction, round(confidence, 2), round(agreement_up, 4), round(z, 4)


@app.get("/health")
def health():
    with _cache_lock:
        training_count = len(_training)
    return {
        "status": "ok",
        "service": "deep-forecast-brain",
        "version": "1.1.0",
        "chronos_model": CHRONOS_MODEL,
        "chronos_loaded": _chronos is not None,
        "local_model_cache": len(_cache),
        "local_training_jobs": training_count,
        "device": DEVICE,
    }


@app.post("/warmup")
def warmup(x_internal_token: str | None = Header(default=None)):
    require_internal(x_internal_token)
    get_chronos()
    return {"ok": True, "chronos_model": CHRONOS_MODEL, "loaded": True}


@app.post("/forecast")
def forecast(req: ForecastRequest, x_internal_token: str | None = Header(default=None)):
    require_internal(x_internal_token)
    symbol = req.symbol.upper()
    timeframe = req.timeframe.upper()
    candles = req.candles or load_db(symbol, timeframe)
    if not candles:
        raise HTTPException(422, "no candles supplied and database history is unavailable")
    try:
        df = normalize_candles(candles)
        horizon = min(req.prediction_length, max(3, len(df) // 20))

        # Fast path: Chronos inference is always synchronous.
        chronos, q10, q90 = chronos_predict(df, horizon)

        # Local PatchTST/TFT are research models. Never block the live request for retraining.
        bundle, freshness = get_cached_bundle(symbol, timeframe, df, horizon)
        training_scheduled = False
        if bundle is None or freshness == "STALE":
            training_scheduled = schedule_local_training(symbol, timeframe, df, horizon)

        patch = tft = None
        if bundle is not None:
            try:
                patch, tft = local_predict(bundle, df, horizon)
            except Exception:
                patch = tft = None
    except Exception as exc:
        raise HTTPException(503, f"deep forecast failed: {type(exc).__name__}: {exc}")

    paths = [chronos]
    weights = {"chronos2": 1.0, "patchtst": 0.0, "tft": 0.0}
    ensemble = chronos.copy()
    if patch is not None and tft is not None:
        paths = [chronos, patch, tft]
        weights = {"chronos2": 0.50, "patchtst": 0.25, "tft": 0.25}
        ensemble = 0.50 * chronos + 0.25 * patch + 0.25 * tft

    last_price = float(df["close"].iloc[-1])
    hist_vol = float(df["ret"].tail(100).std()) or 1e-5
    direction, confidence, agreement_up, z = direction_info(paths, ensemble, hist_vol)

    models: dict[str, Any] = {
        "chronos2": {
            "model": CHRONOS_MODEL,
            "status": "READY",
            "forecast_returns": [round(float(x), 8) for x in chronos],
            "forecast_prices": [round(float(x), 6) for x in prices_from_returns(last_price, chronos)],
            "q10_prices": [round(float(x), 6) for x in prices_from_returns(last_price, q10)],
            "q90_prices": [round(float(x), 6) for x in prices_from_returns(last_price, q90)],
        }
    }
    if patch is not None and tft is not None and bundle is not None:
        models["patchtst"] = {
            "model": "PatchTSTMini-trained-on-market-history",
            "status": "READY",
            "forecast_returns": [round(float(x), 8) for x in patch],
            "forecast_prices": [round(float(x), 6) for x in prices_from_returns(last_price, patch)],
            "train_loss": bundle["losses"]["patchtst"],
        }
        models["tft"] = {
            "model": "TemporalFusionTransformerMini-trained-on-market-history",
            "status": "READY",
            "forecast_returns": [round(float(x), 8) for x in tft],
            "forecast_prices": [round(float(x), 6) for x in prices_from_returns(last_price, tft)],
            "train_loss": bundle["losses"]["tft"],
        }
    else:
        state = local_training_status(symbol, timeframe, horizon)
        models["patchtst"] = {"model": "PatchTSTMini-trained-on-market-history", "status": state["state"]}
        models["tft"] = {"model": "TemporalFusionTransformerMini-trained-on-market-history", "status": state["state"]}

    return {
        "status": "READY",
        "symbol": symbol,
        "timeframe": timeframe,
        "prediction_length": horizon,
        "last_price": last_price,
        "direction": direction,
        "direction_confidence": confidence,
        "agreement_up_fraction": agreement_up,
        "normalized_move_strength": z,
        "ensemble_weights": weights,
        "models": models,
        "ensemble_forecast_returns": [round(float(x), 8) for x in ensemble],
        "ensemble_forecast_prices": [round(float(x), 6) for x in prices_from_returns(last_price, ensemble)],
        "local_model_freshness": freshness,
        "local_training_scheduled": training_scheduled,
        "local_training": local_training_status(symbol, timeframe, horizon),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "quality_note": "Direction confidence is an ensemble-strength score, not a calibrated probability. PatchTST/TFT train asynchronously and only join the ensemble when cached models are ready.",
    }
