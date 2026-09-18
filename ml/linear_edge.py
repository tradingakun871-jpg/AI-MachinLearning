import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from ml.performance import trading_metrics


class LinearEdgeRegressor:
    """Regularized linear regression for expected realized R.

    Target is +RR when TP wins and -1R when SL wins. This is intentionally a
    regression model, separate from the probability classifiers.
    """

    def __init__(self,alpha=8.0):
        self.model=Pipeline([
            ("impute",SimpleImputer(strategy="median")),
            ("scale",StandardScaler()),
            ("ridge",Ridge(alpha=float(alpha))),
        ])

    @staticmethod
    def reward_target(y,rr=3.0):
        y=np.asarray(y,dtype=int)
        return np.where(y==1,float(rr),-1.0)

    def fit(self,x,y,rr=3.0):
        self.model.fit(x,self.reward_target(y,rr))
        return self

    def predict(self,x):
        return np.asarray(self.model.predict(x),dtype=float)


def learn_linear_threshold(predicted_r,y,rr=3.0,min_trades=None,min_coverage=0.03):
    pred=np.asarray(predicted_r,dtype=float)
    y=np.asarray(y,dtype=int)
    if len(pred)==0:
        return {"mode":"NONE","threshold":None,"selection":None}

    if min_trades is None:
        min_trades=max(10,int(len(y)*0.03))

    qs=np.quantile(pred,[.35,.45,.55,.65,.72,.78,.84,.88,.92,.95,.97])
    candidates=sorted({round(float(x),4) for x in np.r_[0.0,qs] if np.isfinite(x)})
    rows=[]

    for threshold in candidates:
        mask=pred>=threshold
        trades=int(mask.sum())
        coverage=float(trades/len(y)) if len(y) else 0.0
        if trades<min_trades or coverage<min_coverage:
            continue
        metrics=trading_metrics(y[mask],pred[mask],threshold=-1e9,rr=rr)
        expectancy=float(metrics.get("expectancy_r",-999) or -999)
        pf=metrics.get("profit_factor")
        dd=abs(float(metrics.get("max_drawdown_r",0.0) or 0.0))
        score=expectancy*np.sqrt(trades)-0.01*dd
        row={
            "threshold":float(threshold),
            "coverage":coverage,
            "score":float(score),
            **metrics,
        }
        rows.append(row)

    viable=[
        row for row in rows
        if float(row.get("expectancy_r",-999))>0
        and row.get("profit_factor") is not None
        and float(row.get("profit_factor",0))>=1.05
    ]
    selected=max(viable,key=lambda x:(x["score"],x["trades"])) if viable else None
    return {
        "mode":"LINEAR_EDGE" if selected else "NONE",
        "threshold":None if selected is None else float(selected["threshold"]),
        "selection":selected,
        "candidate_count":len(rows),
    }


def regression_diagnostics(predicted_r,y,rr=3.0):
    pred=np.asarray(predicted_r,dtype=float)
    target=LinearEdgeRegressor.reward_target(y,rr)
    if len(pred)==0:
        return {}
    mae=float(np.mean(np.abs(target-pred)))
    rmse=float(np.sqrt(np.mean((target-pred)**2)))
    corr=None
    if len(pred)>2 and np.std(pred)>0 and np.std(target)>0:
        corr=float(np.corrcoef(pred,target)[0,1])
    return {
        "rows":int(len(pred)),
        "mean_prediction":float(np.mean(pred)),
        "median_prediction":float(np.median(pred)),
        "target_mean":float(np.mean(target)),
        "mae":mae,
        "rmse":rmse,
        "correlation":corr,
    }
