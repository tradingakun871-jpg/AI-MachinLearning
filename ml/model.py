import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression

class ProbabilityEnsemble:
    """LightGBM + XGBoost ensemble with held-out isotonic calibration when calibration has both classes."""
    def __init__(self):
        self.lgbm = LGBMClassifier(
            n_estimators=350, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1
        )
        self.xgb = XGBClassifier(
            n_estimators=350, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, n_jobs=2
        )
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrated = False

    def fit(self, x_train, y_train):
        y=np.asarray(y_train,dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError("training target must contain both classes")
        self.lgbm.fit(x_train, y)
        self.xgb.fit(x_train, y)
        return self

    def raw_probability(self, x):
        p1 = self.lgbm.predict_proba(x)[:, 1]
        p2 = self.xgb.predict_proba(x)[:, 1]
        return 0.5*p1 + 0.5*p2

    def calibrate(self, x_cal, y_cal):
        y=np.asarray(y_cal,dtype=int)
        if len(y) < 25 or len(np.unique(y)) < 2:
            self.calibrated=False
            return self
        self.calibrator.fit(self.raw_probability(x_cal), y)
        self.calibrated = True
        return self

    def predict_proba(self, x):
        raw = self.raw_probability(x)
        return self.calibrator.predict(raw) if self.calibrated else raw


    def feature_importance(self, feature_names):
        """Average normalized native importances from LightGBM and XGBoost."""
        names=list(feature_names)
        l=np.asarray(getattr(self.lgbm,"feature_importances_",np.zeros(len(names))),dtype=float)
        x=np.asarray(getattr(self.xgb,"feature_importances_",np.zeros(len(names))),dtype=float)
        if len(l)!=len(names): l=np.resize(l,len(names))
        if len(x)!=len(names): x=np.resize(x,len(names))
        if l.sum()>0: l=l/l.sum()
        if x.sum()>0: x=x/x.sum()
        avg=(l+x)/2.0
        order=np.argsort(-avg)
        return [
            {
                "feature":names[i],
                "importance":float(avg[i]),
                "lightgbm":float(l[i]),
                "xgboost":float(x[i]),
            }
            for i in order
        ]
