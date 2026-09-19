import asyncio
import json
import os

import ml.auto_train as base_train
import ml.stability as stability_module
from app.database import market_history_status
from ml.auto_train import TrainingManager as BaseTrainingManager
from ml.auto_train_v122 import TrainingManager as V122TrainingManager
from ml.high_winrate_v123 import (
    TARGET_RR,
    evaluate_high_winrate_stability,
    high_winrate_quality_gate,
    learn_high_winrate_policy,
)


XAU_DEEP_HISTORY_REQUIRED={"M3":45000,"M5":27000,"M15":9000}
BTC_DEEP_HISTORY_DAYS=180


def _high_winrate_gate_adapter(
    test_frame,
    test_probability,
    selected_mask,
    applied_thresholds,
    rr,
    policy_status=None,
    temporal_stability=None,
    **_kwargs,
):
    return high_winrate_quality_gate(
        test_frame,
        test_probability,
        selected_mask,
        applied_thresholds,
        rr,
        temporal_stability=temporal_stability,
    )


class TrainingManager(V122TrainingManager):
    """V0.12.4 deep-history + stable-context research trainer."""

    def __init__(self, shadow_service, version="v0.12.4"):
        super().__init__(shadow_service,version=version)
        self.state["XAUUSD"]["required_history"]=dict(XAU_DEEP_HISTORY_REQUIRED)
        self.state["BTCUSD"]["required_history_days"]=BTC_DEEP_HISTORY_DAYS

    async def start_xau(self,force=False):
        if not force and self._restore_existing("XAUUSD"):
            return

        history=market_history_status("XAUUSD")
        enough=all(
            int((history.get(tf) or {}).get("rows",0) or 0)>=minimum
            for tf,minimum in XAU_DEEP_HISTORY_REQUIRED.items()
        )
        if not enough:
            self._update(
                "XAUUSD",
                status="WAITING_FOR_DEEP_MT5_HISTORY",
                history=history,
                required_history=dict(XAU_DEEP_HISTORY_REQUIRED),
                error=None,
            )
            return

        if self.xau_task and not self.xau_task.done():
            return
        if (
            self.task
            and not self.task.done()
            and self.state["BTCUSD"].get("status")!="READY"
        ):
            self._update(
                "XAUUSD",
                status="WAITING_FOR_BTC_TRAINING",
                history=history,
                required_history=dict(XAU_DEEP_HISTORY_REQUIRED),
            )
            return

        self.xau_task=asyncio.create_task(
            self._train_xau(),name="xau-model-training-v0124"
        )

    async def _fit_validate(self,symbol,dataset,rr,horizon,source,history=None):
        old_policy=base_train.learn_threshold_policy
        old_gate=base_train.quality_gate_from_selection
        old_stability_policy=stability_module.learn_threshold_policy
        old_stability_eval=base_train.evaluate_temporal_stability
        try:
            base_train.learn_threshold_policy=learn_high_winrate_policy
            base_train.quality_gate_from_selection=_high_winrate_gate_adapter
            stability_module.learn_threshold_policy=learn_high_winrate_policy
            base_train.evaluate_temporal_stability=evaluate_high_winrate_stability
            return await BaseTrainingManager._fit_validate(
                self,symbol,dataset,TARGET_RR,horizon,source,history
            )
        finally:
            base_train.learn_threshold_policy=old_policy
            base_train.quality_gate_from_selection=old_gate
            stability_module.learn_threshold_policy=old_stability_policy
            base_train.evaluate_temporal_stability=old_stability_eval

    async def _train_btc(self):
        previous=os.environ.get("BTC_TRAIN_DAYS")
        try:
            requested=int(previous) if previous is not None else 0
        except Exception:
            requested=0
        os.environ["BTC_TRAIN_DAYS"]=str(max(BTC_DEEP_HISTORY_DAYS,requested))
        try:
            await super()._train_btc()
        finally:
            if previous is None:
                os.environ.pop("BTC_TRAIN_DAYS",None)
            else:
                os.environ["BTC_TRAIN_DAYS"]=previous

    def _log_training_result(self,symbol):
        state=self.state.get(symbol) or {}
        metrics=state.get("metrics") or {}
        quality=state.get("quality") or {}
        trading=metrics.get("trading") or {}
        temporal=metrics.get("temporal_stability") or {}
        history=metrics.get("history") or state.get("history") or {}
        sides=metrics.get("side_models") or {}
        failed=[
            name for name,item in (quality.get("checks") or {}).items()
            if not item.get("passed",False)
        ]

        def side_summary(side):
            info=sides.get(side) or {}
            threshold=info.get("threshold_policy") or {}
            return {
                "threshold_mode":threshold.get("mode"),
                "stable_context_counts":threshold.get("stable_context_counts"),
                "recovery_used":threshold.get("recovery_used"),
                "hybrid_mode":((info.get("hybrid_gate") or {}).get("mode")),
                "meta_mode":((info.get("meta_precision") or {}).get("mode")),
                "test_selected_rows":info.get("test_selected_rows"),
                "test_coverage":info.get("test_coverage"),
                "test_trading":info.get("test_trading"),
            }

        payload={
            "event":"V0.12.4_TRAINING_RESULT",
            "symbol":symbol,
            "status":state.get("status"),
            "dataset_rows":state.get("dataset_rows"),
            "history":history,
            "required_xau_history":XAU_DEEP_HISTORY_REQUIRED if symbol=="XAUUSD" else None,
            "required_btc_days":BTC_DEEP_HISTORY_DAYS if symbol=="BTCUSD" else None,
            "error":state.get("error"),
            "quality":quality.get("status"),
            "failed_checks":failed,
            "rr":metrics.get("rr"),
            "trades":trading.get("trades"),
            "win_rate":trading.get("win_rate"),
            "net_r":trading.get("net_r"),
            "expectancy_r":trading.get("expectancy_r"),
            "profit_factor":trading.get("profit_factor"),
            "max_drawdown_r":trading.get("max_drawdown_r"),
            "coverage":quality.get("coverage"),
            "temporal_stability":temporal.get("status"),
            "temporal_summary":temporal.get("summary"),
            "high_winrate_stability":temporal.get("high_winrate_stability"),
            "buy":side_summary("BUY"),
            "sell":side_summary("SELL"),
        }
        try:
            print(json.dumps(payload,default=str),flush=True)
        except Exception:
            print(str(payload),flush=True)
