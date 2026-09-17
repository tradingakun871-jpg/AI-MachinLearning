import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression

class ProbabilityEnsemble:
    """LightGBM + XGBoost ensemble with held-out isotonic calibration."""
    def __init__(self):
        self.lgbm = LGBMClassifier(n_estimators=350, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42)
        self.xgb = XGBClassifier(n_estimators=350, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrated = False

    def fit(self, x_train, y_train):
        self.lgbm.fit(x_train, y_train)
        self.xgb.fit(x_train, y_train)
        return self

    def raw_probability(self, x):
        p1 = self.lgbm.predict_proba(x)[:, 1]
        p2 = self.xgb.predict_proba(x)[:, 1]
        return 0.5*p1 + 0.5*p2

    def calibrate(self, x_cal, y_cal):
        self.calibrator.fit(self.raw_probability(x_cal), np.asarray(y_cal))
        self.calibrated = True
        return self

    def predict_proba(self, x):
        raw = self.raw_probability(x)
        return self.calibrator.predict(raw) if self.calibrated else raw
