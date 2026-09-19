import json

from ml.auto_train_v12 import TrainingManager as V121TrainingManager


class TrainingManager(V121TrainingManager):
    """V0.12.2 high-winrate RR 1:2 trainer.

    Uses V0.12.1's hybrid/meta stack with V0.12.2 calibration policy:
    strict confidence first, controlled coverage recovery second, while the
    independent final quality gate remains unchanged and strict.
    """

    def __init__(self, shadow_service, version="v0.12.2"):
        super().__init__(shadow_service, version=version)

    def _log_training_result(self,symbol):
        state=self.state.get(symbol) or {}
        metrics=state.get("metrics") or {}
        quality=state.get("quality") or {}
        trading=metrics.get("trading") or {}
        sides=metrics.get("side_models") or {}
        temporal=metrics.get("temporal_stability") or {}
        failed=[
            name for name,item in (quality.get("checks") or {}).items()
            if not item.get("passed",False)
        ]

        def side_summary(side):
            info=sides.get(side) or {}
            policy=info.get("threshold_policy") or {}
            hybrid=info.get("hybrid_gate") or {}
            meta=info.get("meta_precision") or {}
            return {
                "threshold_mode":policy.get("mode"),
                "recovery_used":policy.get("recovery_used"),
                "global_selection":policy.get("global_selection"),
                "hybrid_mode":hybrid.get("mode"),
                "hybrid_selection":hybrid.get("selection"),
                "meta":meta,
                "test_selected_rows":info.get("test_selected_rows"),
                "test_coverage":info.get("test_coverage"),
                "test_trading":info.get("test_trading"),
            }

        payload={
            "event":"V0.12.2_TRAINING_RESULT",
            "symbol":symbol,
            "status":state.get("status"),
            "dataset_rows":state.get("dataset_rows"),
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
