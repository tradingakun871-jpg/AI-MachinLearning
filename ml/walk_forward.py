import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from ml.model import ProbabilityEnsemble


def walk_forward(frame, features, target="label", min_train=2000, test_size=500, calibration_size=500):
    """Expanding-window evaluation. Calibration is fitted on a held-out block immediately before each test block."""
    results = []
    start = min_train + calibration_size
    while start + test_size <= len(frame):
        train = frame.iloc[:start-calibration_size]
        cal = frame.iloc[start-calibration_size:start]
        test = frame.iloc[start:start+test_size]
        model = ProbabilityEnsemble().fit(train[features], train[target])
        model.calibrate(cal[features], cal[target])
        p = model.predict_proba(test[features])
        y = test[target].to_numpy()
        results.append({
            "start": int(start), "end": int(start+test_size),
            "brier": float(brier_score_loss(y, p)),
            "logloss": float(log_loss(y, np.c_[1-p, p], labels=[0,1])),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
            "mean_probability": float(np.mean(p)), "base_rate": float(np.mean(y))
        })
        start += test_size
    return results
