import json
import numpy as np

import ml.auto_train as base_train
import ml.stability as stability_module

from ml.auto_train import TrainingManager as HybridTrainingManager
from ml.high_winrate import (
    TARGET_RR,
    evaluate_high_winrate_stability,
    high_winrate_quality_gate,
    learn_high_winrate_policy,
)
from ml.hybrid import apply_hybrid_gate
from ml.meta_precision import (
    MetaPrecisionClassifier,
    choose_row_features,
    meta_matrix,
    select_meta_policy,
)
from ml.performance import trading_metrics
from ml.quality import apply_threshold_policy


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


class TrainingManager(HybridTrainingManager):
    """V0.12.1 high-winrate RR 1:2 trainer with confidence-aware meta stacking."""

    def __init__(self, shadow_service, version="v0.12.1"):
        super().__init__(shadow_service, version=version)

    def _restore_existing(self, symbol):
        path=self.registry.path(symbol,self.version)/"model.joblib"
        if not path.exists():
            return False
        try:
            meta=self.registry.metadata(symbol,self.version)
        except Exception:
            return False
        objective=(meta.get("quality_gate") or {}).get("objective") or {}
        side_models=meta.get("side_models") or {}
        if objective.get("name")!="HIGH_WINRATE_RR_1_2":
            return False
        if not side_models or not all(
            (side_models.get(side) or {}).get("meta_precision") is not None
            for side in ("BUY","SELL")
        ):
            return False
        return super()._restore_existing(symbol)

    async def _fit_side(self, side, direction, train, cal, test, features, rr):
        result=await super()._fit_side(side,direction,train,cal,test,features,rr)

        ca=cal[cal["signal_direction"]==direction].copy()
        te=test[test["signal_direction"]==direction].copy()
        bundle=result["bundle"]
        selected_features=bundle.get("features") or []
        cal_p=np.asarray(result["cal_probability"],dtype=float)
        test_p=np.asarray(result["test_probability"],dtype=float)

        meta_metrics={
            "mode":"BYPASS",
            "reason":"not_fitted",
            "fit_rows":0,
            "policy_rows":0,
            "row_features":[],
            "policy":None,
        }

        try:
            linear=bundle.get("linear_model")
            rl=bundle.get("rl_policy")
            if linear is None or rl is None:
                raise RuntimeError("linear/RL components unavailable for meta stack")

            ca_linear=np.asarray(linear.predict(ca[selected_features]),dtype=float)
            te_linear=np.asarray(linear.predict(te[selected_features]),dtype=float)
            ca_state=np.c_[ca[selected_features].to_numpy(dtype=float),ca_linear]
            te_state=np.c_[te[selected_features].to_numpy(dtype=float),te_linear]
            ca_rl_adv=np.asarray(rl.advantage(ca_state),dtype=float)
            te_rl_adv=np.asarray(rl.advantage(te_state),dtype=float)

            row_features=choose_row_features(ca)
            x_cal=meta_matrix(ca,cal_p,ca_linear,ca_rl_adv,row_features)
            x_test=meta_matrix(te,test_p,te_linear,te_rl_adv,row_features)

            split=max(50,min(len(ca)-40,int(len(ca)*0.60)))
            if split<=0 or len(ca)-split<40:
                raise RuntimeError(f"meta calibration split too small: {len(ca)}")
            if len(np.unique(ca.iloc[:split]["label"]))<2:
                raise RuntimeError("meta fit segment contains one class")

            meta=MetaPrecisionClassifier(c=.60)
            meta.fit(x_cal[:split],ca.iloc[:split]["label"])
            cal_meta_p=meta.predict_proba(x_cal)
            test_meta_p=meta.predict_proba(x_test)

            threshold_policy=bundle.get("threshold_policy") or {}
            ca_supervised,_=apply_threshold_policy(ca,cal_p,threshold_policy)

            linear_policy=bundle.get("linear_policy") or {}
            linear_threshold=linear_policy.get("threshold")
            ca_linear_take=(
                ca_linear>=float(linear_threshold)
                if linear_threshold is not None
                else np.zeros(len(ca),dtype=bool)
            )
            ca_rl_take,_=rl.select(ca_state)
            hybrid_mode=(bundle.get("hybrid_gate") or {}).get("mode","NONE")
            ca_base=apply_hybrid_gate(
                hybrid_mode,ca_supervised,ca_linear_take,ca_rl_take
            )

            policy=select_meta_policy(
                ca.iloc[split:]["label"],
                cal_meta_p[split:],
                ca_base[split:],
                rr=rr,
                target_win_rate=.45,
                min_trades=max(10,int((len(ca)-split)*.04)),
                min_coverage=.025,
            )

            if policy.get("mode")=="META_VETO" and policy.get("threshold") is not None:
                test_meta_take=test_meta_p>=float(policy["threshold"])
                result["test_mask"]=np.asarray(result["test_mask"],dtype=bool)&test_meta_take
            else:
                test_meta_take=np.ones(len(te),dtype=bool)

            final_mask=np.asarray(result["test_mask"],dtype=bool)
            final_trade=(
                trading_metrics(
                    te.loc[final_mask,"label"],
                    test_p[final_mask],
                    threshold=0.0,
                    rr=rr,
                )
                if np.any(final_mask) else {"trades":0}
            )

            bundle["meta_precision"]={
                "model":meta,
                "row_features":row_features,
                "policy":policy,
                "fit_rows":int(split),
                "policy_rows":int(len(ca)-split),
            }
            result["metrics"]["test_trading"]=final_trade
            result["metrics"]["test_selected_rows"]=int(final_mask.sum())
            result["metrics"]["test_coverage"]=float(final_mask.mean()) if len(final_mask) else 0.0
            result["metrics"]["meta_precision"]={
                "mode":policy.get("mode"),
                "fit_rows":int(split),
                "policy_rows":int(len(ca)-split),
                "row_features":row_features,
                "policy":policy,
                "test_probability_mean":float(np.mean(test_meta_p)),
                "test_selected_by_meta":int(test_meta_take.sum()),
                "test_selected_final":int(final_mask.sum()),
            }
            meta_metrics=result["metrics"]["meta_precision"]
        except Exception as exc:
            # Meta layer is a precision veto, not a single point of failure.
            # Keep the existing hybrid gate; the independent final gate still
            # decides whether the artifact is allowed into shadow signals.
            bundle["meta_precision"]={
                "model":None,
                "row_features":[],
                "policy":{"mode":"BYPASS","threshold":None},
                "fit_rows":0,
                "policy_rows":0,
            }
            meta_metrics={
                "mode":"BYPASS",
                "reason":f"{type(exc).__name__}: {exc}",
                "fit_rows":0,
                "policy_rows":0,
                "row_features":[],
                "policy":{"mode":"BYPASS","threshold":None},
            }
            result["metrics"]["meta_precision"]=meta_metrics

        return result

    async def _fit_validate(self, symbol, dataset, rr, horizon, source, history=None):
        # The base trainer resolves these helpers from module globals. Swap them
        # only for this training call, then restore them even if training fails.
        old_policy=base_train.learn_threshold_policy
        old_gate=base_train.quality_gate_from_selection
        old_stability_policy=stability_module.learn_threshold_policy
        old_stability_eval=base_train.evaluate_temporal_stability
        try:
            base_train.learn_threshold_policy=learn_high_winrate_policy
            base_train.quality_gate_from_selection=_high_winrate_gate_adapter
            stability_module.learn_threshold_policy=learn_high_winrate_policy
            base_train.evaluate_temporal_stability=evaluate_high_winrate_stability
            return await super()._fit_validate(
                symbol,dataset,TARGET_RR,horizon,source,history
            )
        finally:
            base_train.learn_threshold_policy=old_policy
            base_train.quality_gate_from_selection=old_gate
            stability_module.learn_threshold_policy=old_stability_policy
            base_train.evaluate_temporal_stability=old_stability_eval

    def _log_training_result(self,symbol):
        state=self.state.get(symbol) or {}
        metrics=state.get("metrics") or {}
        quality=state.get("quality") or {}
        trading=metrics.get("trading") or {}
        failed=[
            name for name,item in (quality.get("checks") or {}).items()
            if not item.get("passed",False)
        ]
        payload={
            "event":"V0.12.1_TRAINING_RESULT",
            "symbol":symbol,
            "status":state.get("status"),
            "dataset_rows":state.get("dataset_rows"),
            "error":state.get("error"),
            "quality":quality.get("status"),
            "failed_checks":failed,
            "rr":metrics.get("rr"),
            "trades":trading.get("trades"),
            "win_rate":trading.get("win_rate"),
            "expectancy_r":trading.get("expectancy_r"),
            "profit_factor":trading.get("profit_factor"),
            "max_drawdown_r":trading.get("max_drawdown_r"),
            "coverage":quality.get("coverage"),
            "temporal_stability":(metrics.get("temporal_stability") or {}).get("status"),
            "buy_meta":((metrics.get("side_models") or {}).get("BUY") or {}).get("meta_precision"),
            "sell_meta":((metrics.get("side_models") or {}).get("SELL") or {}).get("meta_precision"),
        }
        try:
            print(json.dumps(payload,default=str),flush=True)
        except Exception:
            print(str(payload),flush=True)

    async def _train_btc(self):
        await super()._train_btc()
        self._log_training_result("BTCUSD")

    async def _train_xau(self):
        await super()._train_xau()
        self._log_training_result("XAUUSD")
