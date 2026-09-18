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
from ml.performance import probability_metrics, trading_metrics
from ml.quality import (
    apply_threshold_policy,
    learn_threshold_policy,
    probability_diagnostics,
    quality_gate_from_selection,
    select_features_by_importance,
)
from ml.regime import RegimeClassifier\nfrom ml.stability import evaluate_temporal_stability


class TrainingManager:
    """V0.11.3 side-specific learning with temporal stability and robust calibration."""

    def __init__(self, shadow_service, version="v0.11.3"):
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
        self._update(
            symbol,
            status="READY",
            version=self.version,
            finished_at=meta.get("trained_at"),
            dataset_rows=int(meta.get("dataset_rows") or 0),
            metrics={k: v for k, v in meta.items() if k not in ("trained_at", "mode")},
            quality=meta.get("quality_gate"),
            error=None,
        )
        return True

    async def start_btc(self, force=False):
        if not force and self._restore_existing("BTCUSD"):
            return
        if self.task and not self.task.done():
            return
        self.task = asyncio.create_task(
            self._train_btc(), name="btc-model-training-v0113"
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
            self._update("XAUUSD", status="WAITING_FOR_MT5_HISTORY", history=history)
            return

        if self.xau_task and not self.xau_task.done():
            return
        if (
            self.task
            and not self.task.done()
            and self.state["BTCUSD"].get("status") != "READY"
        ):
            self._update("XAUUSD", status="WAITING_FOR_BTC_TRAINING", history=history)
            return

        self.xau_task = asyncio.create_task(
            self._train_xau(), name="xau-model-training-v0113"
        )

    @staticmethod
    def _merge_importance(side_metrics):
        merged = {}
        total_weight = 0.0
        for info in side_metrics.values():
            weight = float(info.get("train_rows", 0) or 0)
            total_weight += weight
            for item in info.get("feature_importance", []):
                row = merged.setdefault(
                    item["feature"],
                    {"feature": item["feature"], "importance": 0.0},
                )
                row["importance"] += weight * float(item.get("importance", 0.0))
        if total_weight > 0:
            for row in merged.values():
                row["importance"] /= total_weight
        return sorted(merged.values(), key=lambda x: -x["importance"])

    async def _fit_side(self, side, direction, train, cal, test, features, rr):
        tr = train[train["signal_direction"] == direction].copy()
        ca = cal[cal["signal_direction"] == direction].copy()
        te = test[test["signal_direction"] == direction].copy()

        if min(len(tr), len(ca), len(te)) < 100:
            raise RuntimeError(
                f"{side} split too small: train={len(tr)} cal={len(ca)} test={len(te)}"
            )
        if any(len(np.unique(frame["label"])) < 2 for frame in (tr, ca, te)):
            raise RuntimeError(f"{side} train/cal/test must contain wins and losses")

        scout = ProbabilityEnsemble()
        await asyncio.to_thread(scout.fit, tr[features], tr["label"])
        scout_importance = scout.feature_importance(features)
        selected_features = select_features_by_importance(
            scout_importance,
            tr,
            min_features=14,
            max_features=26,
            cumulative=0.90,
        )
        if len(selected_features) < 10:
            selected_features = features[: min(24, len(features))]

        model = ProbabilityEnsemble()
        await asyncio.to_thread(model.fit, tr[selected_features], tr["label"])
        await asyncio.to_thread(model.calibrate, ca[selected_features], ca["label"])

        cal_p = model.predict_proba(ca[selected_features])
        policy = learn_threshold_policy(ca, cal_p, rr=rr)

        test_p = model.predict_proba(te[selected_features])
        test_take, test_thresholds = apply_threshold_policy(te, test_p, policy)
        test_trade = (
            trading_metrics(
                te.loc[test_take, "label"],
                test_p[test_take],
                threshold=0.0,
                rr=rr,
            )
            if np.any(test_take)
            else {"trades": 0}
        )
        test_prob = probability_metrics(te["label"], test_p)

        final_importance = model.feature_importance(selected_features)
        return {
            "bundle": {
                "model": model,
                "features": selected_features,
                "threshold_policy": policy,
                "enabled": policy.get("mode") != "NONE",
                "side": side,
            },
            "cal_probability": cal_p,
            "test_probability": test_p,
            "test_mask": test_take,
            "test_thresholds": test_thresholds,
            "test_index": te.index.to_numpy(),
            "metrics": {
                "train_rows": int(len(tr)),
                "calibration_rows": int(len(ca)),
                "test_rows": int(len(te)),
                "selected_features": selected_features,
                "selected_feature_count": int(len(selected_features)),
                "feature_importance": final_importance[:20],
                "threshold_policy": policy,
                "calibration_probability_diagnostics": probability_diagnostics(
                    ca["label"], cal_p, ca
                ),
                "test_probability_diagnostics": probability_diagnostics(
                    te["label"], test_p, te
                ),
                "test_probability": test_prob,
                "test_trading": test_trade,
                "test_selected_rows": int(np.sum(test_take)),
                "test_coverage": float(np.mean(test_take)) if len(test_take) else 0.0,
            },
        }

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

        development = dataset.iloc[:test_pos].copy()
        development = development[
            development["source_index"] < test_source - horizon
        ].copy()

        self._update(symbol, status="TEMPORAL_WALK_FORWARD")
        temporal_stability = await asyncio.to_thread(
            evaluate_temporal_stability,
            development,
            FEATURE_COLUMNS,
            rr,
            horizon,
            3,
        )

        self._update(symbol, status="TRAINING_REGIME")
        regime = await asyncio.to_thread(RegimeClassifier().fit, train)
        for frame in (train, cal, test):
            frame["regime"] = regime.predict(frame)

        features = FEATURE_COLUMNS + ["regime"]
        missing = [name for name in features if name not in train.columns]
        if missing:
            raise RuntimeError(f"missing training features: {missing}")

        self._update(symbol, status="FEATURE_PRUNING")
        self._update(symbol, status="TRAINING_SIDE_MODELS")

        buy = await self._fit_side("BUY", 1, train, cal, test, features, rr)
        sell = await self._fit_side("SELL", -1, train, cal, test, features, rr)
        side_results = {"BUY": buy, "SELL": sell}

        self._update(symbol, status="LEARNING_CONTEXT_POLICY")

        combined_probability = np.full(len(test), np.nan, dtype=float)
        combined_take = np.zeros(len(test), dtype=bool)
        combined_thresholds = np.full(len(test), np.nan, dtype=float)

        index_position = {idx: pos for pos, idx in enumerate(test.index.to_numpy())}
        for info in side_results.values():
            for local_pos, idx in enumerate(info["test_index"]):
                pos = index_position[idx]
                combined_probability[pos] = info["test_probability"][local_pos]
                combined_take[pos] = bool(info["test_mask"][local_pos])
                combined_thresholds[pos] = info["test_thresholds"][local_pos]

        if not np.isfinite(combined_probability).all():
            raise RuntimeError("side-specific probability coverage is incomplete")

        self._update(symbol, status="VALIDATING_QUALITY")
        gate = quality_gate_from_selection(
            test,
            combined_probability,
            combined_take,
            combined_thresholds,
            rr,
            policy_status="SIDE_CONTEXTUAL",
        )

        side_metrics = {
            side: info["metrics"] for side, info in side_results.items()
        }
        aggregate_importance = self._merge_importance(side_metrics)

        threshold_policy = {
            "mode": "SIDE_CONTEXTUAL",
            "BUY": buy["bundle"]["threshold_policy"],
            "SELL": sell["bundle"]["threshold_policy"],
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
            "side_models": side_metrics,
            "threshold_policy": threshold_policy,
            "feature_importance": aggregate_importance[:25],
            "probability": gate["probability"],
            "probability_diagnostics": gate["probability_diagnostics"],
            "trading": gate["trading"],
            "quality_gate": gate,
            "calibration_used": bool(
                buy["bundle"]["model"].calibrated
                and sell["bundle"]["model"].calibrated
            ),
        }

        bundle = {
            "regime": regime,
            "side_models": {
                "BUY": buy["bundle"],
                "SELL": sell["bundle"],
            },
            "rr": rr,
            "symbol": symbol,
            "version": self.version,
            "threshold_policy": threshold_policy,
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
            days = max(60, int(os.getenv("BTC_TRAIN_DAYS", "90")))
            horizon = max(20, int(os.getenv("BTC_LABEL_HORIZON", "80")))
            rr = float(os.getenv("DEFAULT_RR", "3.0"))

            feed = CoinbaseBTCFeed(self.shadow, poll_seconds=60)
            async with httpx.AsyncClient(
                base_url=feed.BASE_URL,
                timeout=30.0,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "AI-Market-Intelligence-Trainer/0.11.3",
                },
            ) as client:
                m1 = await feed._fetch_candles(client, 60, days * 24 * 60)
                m5 = await feed._fetch_candles(client, 300, days * 24 * 12)
                m15 = await feed._fetch_candles(client, 900, days * 24 * 4)

            m3 = feed._resample_m3(m1)
            if len(m3) < 15000 or len(m5) < 8000 or len(m15) < 2500:
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
                    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

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
