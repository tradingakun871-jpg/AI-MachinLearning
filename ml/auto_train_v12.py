import ml.auto_train as base_train
import ml.stability as stability_module

from ml.auto_train import TrainingManager as HybridTrainingManager
from ml.high_winrate import (
    TARGET_RR,
    high_winrate_quality_gate,
    learn_high_winrate_policy,
)


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
    """V0.12 high-winrate RR 1:2 research trainer.

    It reuses the V0.11.4 hybrid LightGBM/XGBoost + linear regression + offline
    Q-learning pipeline, but swaps in a calibration policy that explicitly
    targets higher precision at RR 1:2 and a stricter independent holdout gate.
    """

    def __init__(self, shadow_service, version="v0.12.0"):
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
        if objective.get("name")!="HIGH_WINRATE_RR_1_2":
            return False
        return super()._restore_existing(symbol)

    async def _fit_validate(self, symbol, dataset, rr, horizon, source, history=None):
        # The base trainer resolves these helpers from module globals. Swap them
        # only for this training call, then restore them even if training fails.
        old_policy=base_train.learn_threshold_policy
        old_gate=base_train.quality_gate_from_selection
        old_stability_policy=stability_module.learn_threshold_policy
        try:
            base_train.learn_threshold_policy=learn_high_winrate_policy
            base_train.quality_gate_from_selection=_high_winrate_gate_adapter
            stability_module.learn_threshold_policy=learn_high_winrate_policy
            return await super()._fit_validate(
                symbol,dataset,TARGET_RR,horizon,source,history
            )
        finally:
            base_train.learn_threshold_policy=old_policy
            base_train.quality_gate_from_selection=old_gate
            stability_module.learn_threshold_policy=old_stability_policy
