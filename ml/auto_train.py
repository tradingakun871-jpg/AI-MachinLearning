import asyncio
import os
from datetime import datetime, timezone

import httpx
import numpy as np
from sklearn.metrics import average_precision_score

from data.coinbase import CoinbaseBTCFeed
from data.dataset import build
from ml.features import FEATURE_COLUMNS
from ml.model import ProbabilityEnsemble
from ml.model_registry import ModelRegistry
from ml.performance import probability_metrics, trading_metrics
from ml.regime import RegimeClassifier
from ml.thresholds import threshold_sweep


class TrainingManager:
    """Background research trainer. Never submits broker orders."""

    def __init__(self, shadow_service, version="v0.9"):
        self.shadow = shadow_service
        self.version = version
        self.registry = ModelRegistry()
        self.task = None
        self.state = {
            "BTCUSD": {
                "status": "IDLE",
                "version": version,
                "started_at": None,
                "finished_at": None,
                "error": None,
                "dataset_rows": 0,
                "metrics": None,
            },
            "XAUUSD": {
                "status": "WAITING_FOR_MT5_HISTORY",
                "version": version,
                "started_at": None,
                "finished_at": None,
                "error": None,
                "dataset_rows": 0,
                "metrics": None,
            },
        }

    def status(self):
        return self.state

    async def start_btc(self, force=False):
        existing = self.registry.path("BTCUSD", self.version) / "model.joblib"
        if existing.exists() and not force:
            self.state["BTCUSD"]["status"] = "READY"
            return
        if self.task and not self.task.done():
            return
        self.task = asyncio.create_task(self._train_btc(), name="btc-model-training")

    async def _train_btc(self):
        s = self.state["BTCUSD"]
        s.update({
            "status": "DOWNLOADING_HISTORY",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "error": None,
            "dataset_rows": 0,
            "metrics": None,
        })

        try:
            days = max(7, int(os.getenv("BTC_TRAIN_DAYS", "30")))
            horizon = max(20, int(os.getenv("BTC_LABEL_HORIZON", "80")))
            rr = float(os.getenv("DEFAULT_RR", "3.0"))

            feed = CoinbaseBTCFeed(self.shadow, poll_seconds=60)
            m1_points = days * 24 * 60
            m5_points = days * 24 * 12
            m15_points = days * 24 * 4

            async with httpx.AsyncClient(
                base_url=feed.BASE_URL,
                timeout=30.0,
                headers={"Accept": "application/json", "User-Agent": "AI-Market-Intelligence-Trainer/0.9"},
            ) as client:
                m1 = await feed._fetch_candles(client, 60, m1_points)
                m5 = await feed._fetch_candles(client, 300, m5_points)
                m15 = await feed._fetch_candles(client, 900, m15_points)

            m3 = feed._resample_m3(m1)
            if len(m3) < 2500 or len(m5) < 1000 or len(m15) < 500:
                raise RuntimeError(
                    f"insufficient Coinbase history: M3={len(m3)} M5={len(m5)} M15={len(m15)}"
                )

            s["status"] = "BUILDING_DATASET"
            dataset = await asyncio.to_thread(build, m3, m5, m15, rr, horizon)
            dataset = dataset.sort_values("timestamp").reset_index(drop=True)
            s["dataset_rows"] = int(len(dataset))

            if len(dataset) < 1000:
                raise RuntimeError(f"insufficient resolved SMC training candidates: {len(dataset)}")

            # Time-only split with source-bar purge so an 80-bar label horizon cannot cross split boundaries.
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

            if min(len(train), len(cal), len(test)) < 100:
                raise RuntimeError(
                    f"split too small after purge: train={len(train)} cal={len(cal)} test={len(test)}"
                )

            if len(np.unique(train["label"])) < 2 or len(np.unique(test["label"])) < 2:
                raise RuntimeError("training/test split must contain wins and losses")

            s["status"] = "TRAINING_REGIME"
            regime = await asyncio.to_thread(RegimeClassifier().fit, train)

            for frame in (train, cal, test):
                frame["regime"] = regime.predict(frame)

            features = FEATURE_COLUMNS + ["regime"]

            s["status"] = "TRAINING_ENSEMBLE"
            model = ProbabilityEnsemble()
            await asyncio.to_thread(model.fit, train[features], train["label"])
            await asyncio.to_thread(model.calibrate, cal[features], cal["label"])

            cal_p = model.predict_proba(cal[features])
            sweep = threshold_sweep(cal["label"], cal_p, rr=rr, min_trades=max(30, int(len(cal)*0.08)))
            viable = [x for x in sweep if x.get("expectancy_r", -999) > 0 and x.get("net_r", -999) > 0]
            selected = max(viable, key=lambda x: (x.get("expectancy_r", -999), x.get("trades", 0))) if viable else None
            threshold = float(selected["threshold"]) if selected else 0.60

            s["status"] = "VALIDATING"
            test_p = model.predict_proba(test[features])
            prob = probability_metrics(test["label"], test_p)
            prob["pr_auc"] = float(average_precision_score(test["label"], test_p))
            trade = trading_metrics(test["label"], test_p, threshold=threshold, rr=rr)

            metrics = {
                "source": "Coinbase Exchange BTC-USD",
                "days": days,
                "rr": rr,
                "horizon_m3_bars": horizon,
                "dataset_rows": int(len(dataset)),
                "train_rows": int(len(train)),
                "calibration_rows": int(len(cal)),
                "test_rows": int(len(test)),
                "test_base_rate": float(test["label"].mean()),
                "selected_threshold_from_calibration": threshold,
                "probability": prob,
                "trading": trade,
                "calibration_used": bool(model.calibrated),
            }

            bundle = {
                "model": model,
                "regime": regime,
                "features": features,
                "threshold": threshold,
                "rr": rr,
                "symbol": "BTCUSD",
                "version": self.version,
            }

            self.registry.save(
                "BTCUSD",
                self.version,
                bundle,
                {
                    **metrics,
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                    "mode": "SHADOW_RESEARCH",
                },
            )

            s.update({
                "status": "READY",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics,
            })

        except asyncio.CancelledError:
            s["status"] = "CANCELLED"
            raise
        except Exception as exc:
            s.update({
                "status": "FAILED",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            })
