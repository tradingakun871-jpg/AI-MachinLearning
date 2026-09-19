import asyncio
import json
import os

import numpy as np

import ml.auto_train as base_train
import ml.stability as stability_module
from app.database import market_history_status
from ml.auto_train import TrainingManager as BaseTrainingManager
from ml.auto_train_v122 import TrainingManager as V122TrainingManager
from ml.high_winrate_v125 import (
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


def _restore_env(name,previous):
    if previous is None:
        os.environ.pop(name,None)
    else:
        os.environ[name]=previous


class TrainingManager(V122TrainingManager):
    """V0.12.5 deep-history trainer with directional walk-forward gating.

    RR is forced to 1:2 for BOTH label construction and downstream validation.
    BUY/SELL activation is decided only from pre-holdout walk-forward evidence.
    The untouched final holdout never decides whether a direction is enabled.
    """

    def __init__(self, shadow_service, version="v0.12.5"):
        super().__init__(shadow_service,version=version)
        self.state["XAUUSD"]["required_history"]=dict(XAU_DEEP_HISTORY_REQUIRED)
        self.state["BTCUSD"]["required_history_days"]=BTC_DEEP_HISTORY_DAYS
        self.state["BTCUSD"]["label_rr"]=TARGET_RR
        self.state["XAUUSD"]["label_rr"]=TARGET_RR
        self._directional_stability={}

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
                label_rr=TARGET_RR,
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
                label_rr=TARGET_RR,
            )
            return

        self.xau_task=asyncio.create_task(
            self._train_xau(),name="xau-model-training-v0125"
        )

    async def _fit_side(self,side,direction,train,cal,test,features,rr):
        result=await super()._fit_side(side,direction,train,cal,test,features,rr)
        directional=(self._directional_stability or {}).get(side) or {}
        directional_pass=bool(directional.get("passed",False))

        result["bundle"]["directional_stability"]=directional
        result["bundle"]["directional_gate_passed"]=directional_pass
        result["bundle"]["enabled"]=bool(
            result["bundle"].get("enabled",False) and directional_pass
        )
        result["metrics"]["directional_stability"]=directional
        result["metrics"]["directional_gate_passed"]=directional_pass

        # The side gate is development-only. If a direction is unstable, do not
        # let it contribute any final-holdout trades. Probabilities are retained
        # for diagnostics, but selection is forced to SKIP.
        if not directional_pass:
            result["test_mask"]=np.zeros(len(result["test_mask"]),dtype=bool)
            result["metrics"]["test_selected_rows"]=0
            result["metrics"]["test_coverage"]=0.0
            result["metrics"]["test_trading"]={"trades":0}

        return result

    async def _fit_validate(self,symbol,dataset,rr,horizon,source,history=None):
        # Ignore inherited/environment RR here as a second guard. The dataset is
        # built under the same TARGET_RR in _train_btc/_train_xau below.
        rr=TARGET_RR
        old_policy=base_train.learn_threshold_policy
        old_gate=base_train.quality_gate_from_selection
        old_stability_policy=stability_module.learn_threshold_policy
        old_stability_eval=base_train.evaluate_temporal_stability
        self._directional_stability={}

        def capture_stability(*args,**kwargs):
            result=evaluate_high_winrate_stability(*args,**kwargs)
            self._directional_stability=result.get("directional_stability") or {}
            return result

        try:
            base_train.learn_threshold_policy=learn_high_winrate_policy
            base_train.quality_gate_from_selection=_high_winrate_gate_adapter
            stability_module.learn_threshold_policy=learn_high_winrate_policy
            base_train.evaluate_temporal_stability=capture_stability
            return await BaseTrainingManager._fit_validate(
                self,symbol,dataset,TARGET_RR,horizon,source,history
            )
        finally:
            base_train.learn_threshold_policy=old_policy
            base_train.quality_gate_from_selection=old_gate
            stability_module.learn_threshold_policy=old_stability_policy
            base_train.evaluate_temporal_stability=old_stability_eval
            self._directional_stability={}

    async def _train_btc(self):
        previous_days=os.environ.get("BTC_TRAIN_DAYS")
        previous_rr=os.environ.get("DEFAULT_RR")
        try:
            requested=int(previous_days) if previous_days is not None else 0
        except Exception:
            requested=0

        os.environ["BTC_TRAIN_DAYS"]=str(max(BTC_DEEP_HISTORY_DAYS,requested))
        os.environ["DEFAULT_RR"]=str(TARGET_RR)
        self.state["BTCUSD"]["label_rr"]=TARGET_RR
        try:
            await super()._train_btc()
        finally:
            _restore_env("BTC_TRAIN_DAYS",previous_days)
            _restore_env("DEFAULT_RR",previous_rr)

    async def _train_xau(self):
        previous_rr=os.environ.get("DEFAULT_RR")
        os.environ["DEFAULT_RR"]=str(TARGET_RR)
        self.state["XAUUSD"]["label_rr"]=TARGET_RR
        try:
            await super()._train_xau()
        finally:
            _restore_env("DEFAULT_RR",previous_rr)

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
                "directional_gate_passed":info.get("directional_gate_passed"),
                "directional_stability":info.get("directional_stability"),
                "recovery_used":threshold.get("recovery_used"),
                "hybrid_mode":((info.get("hybrid_gate") or {}).get("mode")),
                "meta_mode":((info.get("meta_precision") or {}).get("mode")),
                "test_selected_rows":info.get("test_selected_rows"),
                "test_coverage":info.get("test_coverage"),
                "test_trading":info.get("test_trading"),
            }

        payload={
            "event":"V0.12.5_TRAINING_RESULT",
            "symbol":symbol,
            "status":state.get("status"),
            "dataset_rows":state.get("dataset_rows"),
            "history":history,
            "required_xau_history":XAU_DEEP_HISTORY_REQUIRED if symbol=="XAUUSD" else None,
            "required_btc_days":BTC_DEEP_HISTORY_DAYS if symbol=="BTCUSD" else None,
            "label_rr":TARGET_RR,
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
            "directional_stability":temporal.get("directional_stability"),
            "stable_sides":temporal.get("stable_sides"),
            "buy":side_summary("BUY"),
            "sell":side_summary("SELL"),
        }
        try:
            print(json.dumps(payload,default=str),flush=True)
        except Exception:
            print(str(payload),flush=True)
