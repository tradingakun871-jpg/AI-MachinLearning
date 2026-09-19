import numpy as np
from sklearn.metrics import average_precision_score

from ml.model import ProbabilityEnsemble
from ml.performance import probability_metrics, trading_metrics
from ml.quality import apply_threshold_policy, learn_threshold_policy
from ml.regime import RegimeClassifier


def _finite_values(values):
    out=[]
    for value in values:
        if value is None:
            continue
        try:
            number=float(value)
        except (TypeError,ValueError):
            continue
        if np.isfinite(number):
            out.append(number)
    return out


def _median(values):
    vals=_finite_values(values)
    return float(np.median(vals)) if vals else None


def _minimum(values):
    vals=_finite_values(values)
    return float(min(vals)) if vals else None


def evaluate_temporal_stability(
    development,
    base_features,
    rr=3.0,
    horizon=80,
    folds=3,
    min_side_train=120,
    n_estimators=140,
):
    """Expanding-window walk-forward diagnostic using only pre-test development data.

    Each fold independently fits regime + BUY/SELL models on past data,
    calibrates on a later chronological slice, and validates on the next slice.
    The final holdout test is not touched here.
    """
    dev=development.sort_values("timestamp").reset_index(drop=True).copy()
    n=len(dev)
    if n<1400:
        return {
            "status":"INSUFFICIENT_DATA",
            "passed":False,
            "folds":[],
            "reason":f"development rows={n}",
        }

    initial=max(700,int(n*0.46))
    remaining=n-initial
    window=max(180,remaining//folds)
    fold_rows=[]

    for fold in range(folds):
        val_start=initial+fold*window
        val_end=n if fold==folds-1 else min(n,val_start+window)
        if val_start>=n or val_end-val_start<120:
            continue

        prefix=dev.iloc[:val_start].copy()
        val=dev.iloc[val_start:val_end].copy()

        cal_pos=max(300,int(len(prefix)*0.80))
        if cal_pos>=len(prefix)-80:
            continue

        cal_source=int(prefix.iloc[cal_pos]["source_index"])
        val_source=int(val.iloc[0]["source_index"])

        fit=prefix.iloc[:cal_pos].copy()
        fit=fit[fit["source_index"]<cal_source-horizon].copy()

        cal=prefix.iloc[cal_pos:].copy()
        cal=cal[cal["source_index"]<val_source-horizon].copy()

        if min(len(fit),len(cal),len(val))<100:
            continue
        if any(len(np.unique(frame["label"]))<2 for frame in (fit,cal,val)):
            continue

        regime=RegimeClassifier().fit(fit)
        for frame in (fit,cal,val):
            frame["regime"]=regime.predict(frame)

        features=list(base_features)+["regime"]
        combined_p=np.full(len(val),np.nan,dtype=float)
        combined_take=np.zeros(len(val),dtype=bool)
        side_detail={}
        valid=True

        for direction,side in ((1,"BUY"),(-1,"SELL")):
            tr=fit[fit["signal_direction"]==direction].copy()
            ca=cal[cal["signal_direction"]==direction].copy()
            ve=val[val["signal_direction"]==direction].copy()

            if min(len(tr),len(ca),len(ve))<50 or len(tr)<min_side_train:
                valid=False
                break
            if any(len(np.unique(frame["label"]))<2 for frame in (tr,ca,ve)):
                valid=False
                break

            model=ProbabilityEnsemble(n_estimators=n_estimators)
            model.fit(tr[features],tr["label"])
            model.calibrate(ca[features],ca["label"],method="auto")
            cal_p=model.predict_proba(ca[features])
            policy=learn_threshold_policy(ca,cal_p,rr=rr)
            val_p=model.predict_proba(ve[features])
            take,_=apply_threshold_policy(ve,val_p,policy)

            positions=val.index.get_indexer(ve.index)
            combined_p[positions]=val_p
            combined_take[positions]=take

            side_trade=(
                trading_metrics(ve.loc[take,"label"],val_p[take],threshold=0.0,rr=rr)
                if np.any(take) else {"trades":0}
            )
            side_detail[side]={
                "train_rows":int(len(tr)),
                "calibration_rows":int(len(ca)),
                "validation_rows":int(len(ve)),
                "policy_mode":policy.get("mode"),
                "calibration_method":model.calibration_method,
                "selected_rows":int(np.sum(take)),
                "trading":side_trade,
            }

        if not valid or not np.isfinite(combined_p).all():
            continue

        prob=probability_metrics(val["label"],combined_p)
        prob["pr_auc"]=float(average_precision_score(val["label"],combined_p))
        trade=(
            trading_metrics(
                val.loc[combined_take,"label"],
                combined_p[combined_take],
                threshold=0.0,
                rr=rr,
            )
            if np.any(combined_take) else {"trades":0}
        )
        trades=int(trade.get("trades",0) or 0)
        expectancy=float(trade.get("expectancy_r",-999) or -999)
        pf=trade.get("profit_factor")
        auc=prob.get("auc")
        fold_pass=bool(
            trades>=15
            and expectancy>0
            and pf is not None and float(pf)>=1.0
            and auc is not None and float(auc)>=0.50
        )

        fold_rows.append({
            "fold":len(fold_rows)+1,
            "train_rows":int(len(fit)),
            "calibration_rows":int(len(cal)),
            "validation_rows":int(len(val)),
            "validation_start":str(val.iloc[0]["timestamp"]),
            "validation_end":str(val.iloc[-1]["timestamp"]),
            "probability":prob,
            "trading":trade,
            "coverage":float(np.mean(combined_take)),
            "passed":fold_pass,
            "sides":side_detail,
        })

    if len(fold_rows)<3:
        return {
            "status":"INSUFFICIENT_FOLDS",
            "passed":False,
            "folds":fold_rows,
            "fold_count":int(len(fold_rows)),
        }

    profitable=sum(
        1 for row in fold_rows
        if float(row["trading"].get("expectancy_r",-999) or -999)>0
    )
    passed_folds=sum(1 for row in fold_rows if row["passed"])
    aucs=[row["probability"].get("auc") for row in fold_rows]
    pfs=[row["trading"].get("profit_factor") for row in fold_rows]
    exps=[row["trading"].get("expectancy_r") for row in fold_rows]
    dds=[row["trading"].get("max_drawdown_r") for row in fold_rows]
    zero_trade_folds=sum(
        1 for row in fold_rows
        if int((row.get("trading") or {}).get("trades",0) or 0)==0
    )

    summary={
        "fold_count":int(len(fold_rows)),
        "passed_folds":int(passed_folds),
        "positive_expectancy_folds":int(profitable),
        "zero_trade_folds":int(zero_trade_folds),
        "median_auc":_median(aucs),
        "median_profit_factor":_median(pfs),
        "median_expectancy_r":_median(exps),
        "worst_drawdown_r":_minimum(dds),
    }

    passed=bool(
        len(fold_rows)>=3
        and profitable>=2
        and summary["median_auc"] is not None and summary["median_auc"]>=0.50
        and summary["median_profit_factor"] is not None and summary["median_profit_factor"]>=1.0
        and summary["median_expectancy_r"] is not None and summary["median_expectancy_r"]>0
    )
    return {
        "status":"PASSED" if passed else "FAILED",
        "passed":passed,
        "summary":summary,
        "folds":fold_rows,
    }
