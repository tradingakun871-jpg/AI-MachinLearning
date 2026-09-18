import ml.auto_train as base_train

from ml.auto_train import TrainingManager as HybridTrainingManager
from ml.high_winrate import (
    TARGET_RR,
    high_winrate_quality_gate,
    learn_high_winrate_policy,
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
        try:
            base_train.learn_threshold_policy=learn_high_winrate_policy
            base_train.quality_gate_from_selection=high_winrate_quality_gate
            return await super()._fit_validate(
                symbol,dataset,TARGET_RR,horizon,source,history
            )
        finally:
            base_train.learn_threshold_policy=old_policy
            base_train.quality_gate_from_selection=old_gate
