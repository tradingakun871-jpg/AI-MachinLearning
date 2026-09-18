import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


def _safe_logloss(y,p):
    p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return float(log_loss(np.asarray(y,dtype=int),np.c_[1-p,p],labels=[0,1]))


class ProbabilityEnsemble:
    """LightGBM + XGBoost ensemble with time-ordered automatic probability calibration."""

    def __init__(self,n_estimators=350):
        n=int(n_estimators)
        self.lgbm=LGBMClassifier(
            n_estimators=n,learning_rate=0.03,num_leaves=31,
            subsample=0.8,colsample_bytree=0.8,random_state=42,verbosity=-1
        )
        self.xgb=XGBClassifier(
            n_estimators=n,max_depth=5,learning_rate=0.03,
            subsample=0.8,colsample_bytree=0.8,eval_metric="logloss",
            random_state=42,n_jobs=2
        )
        self.calibrator=None
        self.calibrated=False
        self.calibration_method="raw"
        self.calibration_diagnostics={}

    def fit(self,x_train,y_train):
        y=np.asarray(y_train,dtype=int)
        if len(np.unique(y))<2:
            raise ValueError("training target must contain both classes")
        self.lgbm.fit(x_train,y)
        self.xgb.fit(x_train,y)
        return self

    def raw_probability(self,x):
        p1=self.lgbm.predict_proba(x)[:,1]
        p2=self.xgb.predict_proba(x)[:,1]
        return 0.5*p1+0.5*p2

    @staticmethod
    def _fit_calibrator(method,raw,y):
        raw=np.asarray(raw,dtype=float)
        y=np.asarray(y,dtype=int)
        if method=="isotonic":
            cal=IsotonicRegression(out_of_bounds="clip")
            cal.fit(raw,y)
            return cal
        if method=="sigmoid":
            cal=LogisticRegression(C=1.0,solver="lbfgs",max_iter=1000)
            cal.fit(raw.reshape(-1,1),y)
            return cal
        return None

    @staticmethod
    def _apply_calibrator(method,calibrator,raw):
        raw=np.asarray(raw,dtype=float)
        if method=="isotonic":
            return calibrator.predict(raw)
        if method=="sigmoid":
            return calibrator.predict_proba(raw.reshape(-1,1))[:,1]
        return raw

    def calibrate(self,x_cal,y_cal,method="auto"):
        """Choose calibration using an internal chronological fit/eval split, then refit on all calibration data."""
        y=np.asarray(y_cal,dtype=int)
        raw=self.raw_probability(x_cal)

        if len(y)<80 or len(np.unique(y))<2:
            self.calibrator=None
            self.calibrated=False
            self.calibration_method="raw"
            self.calibration_diagnostics={
                "method":"raw","reason":"insufficient_calibration_rows_or_classes",
                "rows":int(len(y)),
            }
            return self

        if method!="auto":
            chosen=method
            cal=self._fit_calibrator(chosen,raw,y)
            self.calibrator=cal
            self.calibrated=chosen!="raw"
            self.calibration_method=chosen
            self.calibration_diagnostics={"method":chosen,"rows":int(len(y))}
            return self

        split=max(40,min(len(y)-30,int(len(y)*0.65)))
        fit_raw,eval_raw=raw[:split],raw[split:]
        fit_y,eval_y=y[:split],y[split:]

        if len(np.unique(fit_y))<2 or len(np.unique(eval_y))<2:
            split=max(30,min(len(y)-25,int(len(y)*0.55)))
            fit_raw,eval_raw=raw[:split],raw[split:]
            fit_y,eval_y=y[:split],y[split:]

        candidates={}
        raw_eval=np.clip(eval_raw,1e-6,1-1e-6)
        candidates["raw"]={
            "brier":float(brier_score_loss(eval_y,raw_eval)),
            "logloss":_safe_logloss(eval_y,raw_eval),
        }

        for candidate in ("sigmoid","isotonic"):
            try:
                cal=self._fit_calibrator(candidate,fit_raw,fit_y)
                pred=np.clip(self._apply_calibrator(candidate,cal,eval_raw),1e-6,1-1e-6)
                candidates[candidate]={
                    "brier":float(brier_score_loss(eval_y,pred)),
                    "logloss":_safe_logloss(eval_y,pred),
                }
            except Exception as exc:
                candidates[candidate]={
                    "brier":999.0,"logloss":999.0,
                    "error":f"{type(exc).__name__}: {exc}",
                }

        chosen=min(candidates,key=lambda name:(candidates[name]["brier"],candidates[name]["logloss"]))
        self.calibration_method=chosen
        self.calibration_diagnostics={
            "method":chosen,
            "rows":int(len(y)),
            "selection_fit_rows":int(len(fit_y)),
            "selection_eval_rows":int(len(eval_y)),
            "candidates":candidates,
        }

        if chosen=="raw":
            self.calibrator=None
            self.calibrated=False
        else:
            self.calibrator=self._fit_calibrator(chosen,raw,y)
            self.calibrated=True
        return self

    def predict_proba(self,x):
        raw=self.raw_probability(x)
        if not self.calibrated or self.calibrator is None:
            return raw
        return self._apply_calibrator(self.calibration_method,self.calibrator,raw)

    def feature_importance(self,feature_names):
        """Average normalized native importances from LightGBM and XGBoost."""
        names=list(feature_names)
        l=np.asarray(getattr(self.lgbm,"feature_importances_",np.zeros(len(names))),dtype=float)
        x=np.asarray(getattr(self.xgb,"feature_importances_",np.zeros(len(names))),dtype=float)
        if len(l)!=len(names):
            l=np.resize(l,len(names))
        if len(x)!=len(names):
            x=np.resize(x,len(names))
        if l.sum()>0:
            l=l/l.sum()
        if x.sum()>0:
            x=x/x.sum()
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
