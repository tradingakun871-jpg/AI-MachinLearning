import asyncio
import os
from datetime import datetime, timezone

import httpx
import numpy as np
import pandas as pd

from app.database import load_market_candles, market_history_status, save_training_state
from data.coinbase import CoinbaseBTCFeed
from data.dataset import build
from ml.features import FEATURE_COLUMNS
from ml.model import ProbabilityEnsemble
from ml.model_registry import ModelRegistry
from ml.quality import learn_threshold_policy, quality_gate
from ml.regime import RegimeClassifier


class TrainingManager:
    """V0.11.1 adaptive learning + validation loop for BTCUSD and XAUUSD."""

    def __init__(self, shadow_service, version="v0.11.1"):
        self.shadow = shadow_service
        self.version = version
        self.registry = ModelRegistry()
        self.task = None
        self.xau_task = None
        self.state = {
            "BTCUSD": self._blank("IDLE"),
            "XAUUSD": {**self._blank("WAITING_FOR_MT5_HISTORY"), "history": {}},
        }

    def _blank(self, status):
        return {
            "status": status,
            "version": self.version,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "dataset_rows": 0,
            "metrics": None,
            "quality": None,
        }

    def status(self):
        return self.state

    def _persist(self, symbol):
        try:
            save_training_state(symbol, self.state[symbol])
        except Exception:
            pass

    def _update(self, symbol, **values):
        self.state[symbol].update(values)
        self._persist(symbol)

    def _restore_existing(self, symbol):
        path = self.registry.path(symbol, self.version) / "model.joblib"
        if not path.exists():
            return False
        try:
            meta = self.registry.metadata(symbol, self.version)
        except Exception:
            meta = {}
        quality = meta.get("quality_gate")
        self._update(
            symbol,
            status="READY",
            version=self.version,
            finished_at=meta.get("trained_at"),
            dataset_rows=int(meta.get("dataset_rows") or 0),
            metrics={k: v for k, v in meta.items() if k not in ("trained_at", "mode")},
            quality=quality,
            error=None,
        )
        return True

    async def start_btc(self, force=False):
        if not force and self._restore_existing("BTCUSD"):
            return
        if self.task and not self.task.done():
            return
        self.task = asyncio.create_task(
            self._train_btc(), name="btc-model-training-v0111"
        )

    async def start_xau(self, force=False):
        if not force and self._restore_existing("XAUUSD"):
            return

        history = market_history_status("XAUUSD")
        self.state["XAUUSD"]["history"] = history
        enough = (
            history.get("M3", {}).get("rows", 0) >= 10000
            and history.get("M5", {}).get("rows", 0) >= 5000
            and history.get("M15", {}).get("rows", 0) >= 1500
        )
        if not enough:
            self._update(
                "XAUUSD", status="WAITING_FOR_MT5_HISTORY", history=history
            )
            return

        if self.xau_task and not self.xau_task.done():
            return
        if (
            self.task
            and not self.task.done()
            and self.state["BTCUSD"].get("status") != "READY"
        ):
            self._update(
                "XAUUSD", status="WAITING_FOR_BTC_TRAINING", history=history
            )
            return

        self.xau_task = asyncio.create_task(
            self._train_xau(), name="xau-model-training-v0111"
        )

    async def _fit_validate(self, symbol, dataset, rr, horizon, source, history=None):
        dataset = dataset.sort_values("timestamp").reset_index(drop=True)
        if len(dataset) < 1800:
            raise RuntimeError(
                f"insufficient resolved SMC training candidates: {len(dataset)}"
            )

        n = len(dataset)
        cal_pos = int(n * 0.65)
        test_pos = int(n * 0.82)
        cal_source = int(dataset.iloc[cal_pos]["source_index"])
        test_source = int(dataset.iloc[test_pos]["source_index"])

        train = dataset.iloc[:cal_pos].copy()
        train = train[train["source_index"] < cal_source - horizon].copy()

        cal = dataset.iloc[cal_pos:test_pos].copy()
        cal = cal[cal["source_index"] < test_source - horizon].copy()

        test = dataset.iloc[test_pos:].copy()

        if min(len(train), len(cal), len(test)) < 250:
            raise RuntimeError(
                f"split too small after purge: train={len(train)} "
                f"cal={len(cal)} test={len(test)}"
            )
        if any(len(np.unique(frame["label"])) < 2 for frame in (train, cal, test)):
            raise RuntimeError("train/cal/test must each contain wins and losses")

        self._update(symbol, status="TRAINING_REGIME")
        regime = await asyncio.to_thread(RegimeClassifier().fit, train)
        for frame in (train, cal, test):
            frame["regime"] = regime.predict(frame)

        features = FEATURE_COLUMNS + ["regime"]
        missing = [name for name in features if name not in train.columns]
        if missing:
            raise RuntimeError(f"missing training features: {missing}")

        self._update(symbol, status="TRAINING_ENSEMBLE")
        model = ProbabilityEnsemble()
        await asyncio.to_thread(model.fit, train[features], train["label"])
        await asyncio.to_thread(model.calibrate, cal[features], cal["label"])

        self._update(symbol, status="LEARNING_THRESHOLD_POLICY")
        cal_p = model.predict_proba(cal[features])
        policy = learn_threshold_policy(cal, cal_p, rr=rr)

        threshold = policy.get("global_threshold")
        if threshold is None:
            threshold = float(
                policy.get("break_even_probability", 1.0 / (1.0 + rr)) + 0.01
            )

        self._update(symbol, status="VALIDATING_QUALITY")
        test_p = model.predict_proba(test[features])
        gate = quality_gate(test, test_p, rr, policy)

        importance = model.feature_importance(features)
        learned_filters = {
            "allowed_regimes": policy.get("allowed_regimes", []),
            "allowed_sessions": policy.get("allowed_sessions", []),
            "regime_detail": policy.get("regime_detail", {}),
            "session_detail": policy.get("session_detail", {}),
        }

        metrics = {
            "source": source,
            "rr": rr,
            "horizon_m3_bars": horizon,
            "history": history,
            "dataset_rows": int(len(dataset)),
            "train_rows": int(len(train)),
            "calibration_rows": int(len(cal)),
            "test_rows": int(len(test)),
            "selected_threshold_from_calibration": threshold,
            "threshold_policy": policy,
            "learned_filters": learned_filters,
            "feature_importance": importance[:20],
            "probability": gate["probability"],
            "trading": gate["trading"],
            "quality_gate": gate,
            "calibration_used": bool(model.calibrated),
        }

        bundle = {
            "model": model,
            "regime": regime,
            "features": features,
            "threshold": threshold,
            "threshold_policy": policy,
            "rr": rr,
            "symbol": symbol,
            "version": self.version,
            "learned_filters": learned_filters,
            "quality_gate": gate,
        }

        trained_at = datetime.now(timezone.utc).isoformat()
        self.registry.save(
            symbol,
            self.version,
            bundle,
            {**metrics, "trained_at": trained_at, "mode": "SHADOW_RESEARCH"},
        )
        self._update(
            symbol,
            status="READY",
            finished_at=trained_at,
            dataset_rows=int(len(dataset)),
            metrics=metrics,
            quality=gate,
            error=None,
        )
        return gate

    async def _train_btc(self):
        self._update(
            "BTCUSD",
            status="DOWNLOADING_HISTORY",
            started_at=datetime.now(timezone.utc).isoformat(),
            finished_at=None,
            error=None,
            dataset_rows=0,
            metrics=None,
            quality=None,
        )
        try:
            days = max(30, int(os.getenv("BTC_TRAIN_DAYS", "60")))
            horizon = max(20, int(os.getenv("BTC_LABEL_HORIZON", "80")))
            rr = float(os.getenv("DEFAULT_RR", "3.0"))

            feed = CoinbaseBTCFeed(self.shadow, poll_seconds=60)
            async with httpx.AsyncClient(
                base_url=feed.BASE_URL,
                timeout=30.0,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "AI-Market-Intelligence-Trainer/0.11.1",
                },
            ) as client:
                m1 = await feed._fetch_candles(client, 60, days * 24 * 60)
                m5 = await feed._fetch_candles(client, 300, days * 24 * 12)
                m15 = await feed._fetch_candles(client, 900, days * 24 * 4)

            m3 = feed._resample_m3(m1)
            if len(m3) < 10000 or len(m5) < 5000 or len(m15) < 1500:
                raise RuntimeError(
                    f"insufficient Coinbase history: "
                    f"M3={len(m3)} M5={len(m5)} M15={len(m15)}"
                )

            self._update("BTCUSD", status="BUILDING_DATASET")
            dataset = await asyncio.to_thread(build, m3, m5, m15, rr, horizon)
            self._update("BTCUSD", dataset_rows=int(len(dataset)))

            await self._fit_validate(
                "BTCUSD",
                dataset,
                rr,
                horizon,
                source="Coinbase Exchange BTC-USD",
                history={
                    "days": days,
                    "M3": int(len(m3)),
                    "M5": int(len(m5)),
                    "M15": int(len(m15)),
                },
            )
            await self.start_xau(force=False)

        except asyncio.CancelledError:
            self._update("BTCUSD", status="CANCELLED")
            raise
        except Exception as exc:
            self._update(
                "BTCUSD",
                status="FAILED",
                finished_at=datetime.now(timezone.utc).isoformat(),
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _train_xau(self):
        history = market_history_status("XAUUSD")
        self._update(
            "XAUUSD",
            status="BUILDING_DATASET",
            started_at=datetime.now(timezone.utc).isoformat(),
            finished_at=None,
            error=None,
            dataset_rows=0,
            metrics=None,
            quality=None,
            history=history,
        )
        try:
            rr = float(os.getenv("DEFAULT_RR", "3.0"))
            horizon = max(20, int(os.getenv("XAU_LABEL_HORIZON", "80")))

            m3 = pd.DataFrame(load_market_candles("XAUUSD", "M3"))
            m5 = pd.DataFrame(load_market_candles("XAUUSD", "M5"))
            m15 = pd.DataFrame(load_market_candles("XAUUSD", "M15"))
            for frame in (m3, m5, m15):
                if not frame.empty:
                    frame["timestamp"] = pd.to_datetime(
                        frame["timestamp"], utc=True
                    )

            if len(m3) < 10000 or len(m5) < 5000 or len(m15) < 1500:
                raise RuntimeError(
                    f"insufficient MT5 history: "
                    f"M3={len(m3)} M5={len(m5)} M15={len(m15)}"
                )

            dataset = await asyncio.to_thread(build, m3, m5, m15, rr, horizon)
            self._update("XAUUSD", dataset_rows=int(len(dataset)))

            await self._fit_validate(
                "XAUUSD",
                dataset,
                rr,
                horizon,
                source="MT5 XAUUSD historical backfill",
                history=history,
            )

        except asyncio.CancelledError:
            self._update("XAUUSD", status="CANCELLED")
            raise
        except Exception as exc:
            self._update(
                "XAUUSD",
                status="FAILED",
                finished_at=datetime.now(timezone.utc).isoformat(),
                error=f"{type(exc).__name__}: {exc}",
                history=history,
            )
