import math
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.performance import trading_metrics


META_ROW_FEATURE_CANDIDATES=(
    "smc_confluence",
    "price_action_confluence",
    "orderflow_confluence",
    "context_confluence",
    "pa_strength",
    "pa_trend_efficiency",
    "pa_momentum_3_atr",
    "orderflow_strength",
    "orderflow_source_quality",
    "of_delta_ratio",
    "of_cvd_20",
    "htf_alignment",
    "stop_distance_atr",
    "volatility_z",
    "range_atr",
    "dealing_range_position",
    "distance_to_swing_atr",
    "atr_norm",
    "return_3",
    "regime",
)


def wilson_lower_bound(wins,trades,z=1.96):
    trades=int(trades)
    wins=int(wins)
    if trades<=0:
        return 0.0
    p=wins/trades
    z2=float(z)**2
    denom=1.0+z2/trades
    centre=p+z2/(2.0*trades)
    margin=float(z)*math.sqrt((p*(1.0-p)+z2/(4.0*trades))/trades)
    return float((centre-margin)/denom)


class MetaPrecisionClassifier:
    """Small stacking model that learns when base models agree on high-quality trades."""

    def __init__(self,c=0.45):
        # The meta model is a veto layer. Stronger regularization and a larger
        # deep-history sample are preferred over fitting a small calibration pocket.
        self.model=Pipeline([
            ("impute",SimpleImputer(strategy="median")),
            ("scale",StandardScaler()),
            ("logit",LogisticRegression(C=float(c),max_iter=1200,solver="lbfgs")),
        ])

    def fit(self,x,y):
        y=np.asarray(y,dtype=int)
        if len(y)<60 or len(np.unique(y))<2:
            raise ValueError("meta precision fit needs >=60 rows and both classes")
        self.model.fit(np.asarray(x,dtype=float),y)
        return self

    def predict_proba(self,x):
        return np.asarray(self.model.predict_proba(np.asarray(x,dtype=float))[:,1],dtype=float)


def choose_row_features(frame):
    return [name for name in META_ROW_FEATURE_CANDIDATES if name in frame.columns]


def meta_matrix(frame,base_probability,linear_expected_r,rl_advantage,row_features):
    cols=[
        np.asarray(base_probability,dtype=float),
        np.asarray(linear_expected_r,dtype=float),
        np.asarray(rl_advantage,dtype=float),
    ]
    for name in row_features:
        cols.append(np.asarray(frame[name],dtype=float))
    return np.column_stack(cols)


def _candidate_thresholds(p):
    p=np.asarray(p,dtype=float)
    p=p[np.isfinite(p)]
    if not len(p):
        return []
    qs=np.quantile(p,[.30,.40,.50,.60,.68,.75,.80,.85,.89,.92,.95,.97,.99])
    floor=max(.38,min(.65,float(np.quantile(p,.25))))
    upper=min(.95,max(float(np.max(p)),floor))
    grid=np.arange(floor,upper+1e-9,.005)
    return sorted({round(float(v),4) for v in np.r_[qs,grid] if np.isfinite(v) and floor<=v<=.95})


def select_meta_policy(
    y,
    meta_probability,
    base_mask,
    rr=2.0,
    target_win_rate=.45,
    min_trades=20,
    min_coverage=.04,
):
    """Choose a conservative meta veto.

    The veto is allowed only when the calibration support is material and it
    improves statistical confidence or precision over the already-learned base gate.
    """
    y=np.asarray(y,dtype=int)
    p=np.asarray(meta_probability,dtype=float)
    base=np.asarray(base_mask,dtype=bool)
    rows=[]

    base_metrics=(
        trading_metrics(y[base],p[base],threshold=-1e9,rr=rr)
        if np.any(base) else {"trades":0}
    )
    base_trades=int(base_metrics.get("trades",0) or 0)
    base_wr=float(base_metrics.get("win_rate",0.0) or 0.0)
    base_wins=int(round(base_wr*base_trades))
    base_lcb=wilson_lower_bound(base_wins,base_trades)

    sample_floor=max(int(min_trades),min(30,max(16,int(len(y)*.08))))
    coverage_floor=max(float(min_coverage),0.04)

    for threshold in _candidate_thresholds(p):
        mask=base & (p>=float(threshold))
        trades=int(mask.sum())
        coverage=float(trades/len(y)) if len(y) else 0.0
        if trades<sample_floor or coverage<coverage_floor:
            continue
        metrics=trading_metrics(y[mask],p[mask],threshold=-1e9,rr=rr)
        wr=float(metrics.get("win_rate",0.0) or 0.0)
        wins=int(round(wr*trades))
        lcb=wilson_lower_bound(wins,trades)
        expectancy=float(metrics.get("expectancy_r",-999) or -999)
        pf=metrics.get("profit_factor")
        pfv=float(pf) if pf is not None and np.isfinite(pf) else 0.0
        dd=abs(float(metrics.get("max_drawdown_r",0.0) or 0.0))
        target_bonus=max(0.0,wr-float(target_win_rate))*6.0
        confidence_gain=lcb-base_lcb
        score=(
            4.0*lcb
            +2.0*wr
            +2.0*max(confidence_gain,0.0)
            +target_bonus
            +0.8*expectancy
            +0.06*pfv
            +0.05*math.sqrt(trades)
            -0.015*dd
        )
        rows.append({
            "threshold":float(threshold),
            "coverage":coverage,
            "wilson_lcb_95":lcb,
            "confidence_gain_vs_base":float(confidence_gain),
            "score":float(score),
            "target_met":bool(wr>=target_win_rate),
            **metrics,
        })

    break_even=1.0/(1.0+float(rr))
    viable=[
        row for row in rows
        if float(row.get("expectancy_r",-999))>=0.12
        and row.get("profit_factor") is not None
        and float(row.get("profit_factor",0))>=1.15
        and float(row.get("win_rate",0.0) or 0.0)>=max(break_even+0.03,0.38)
    ]
    target=[row for row in viable if bool(row.get("target_met"))]
    pool=target if target else viable
    selected=max(pool,key=lambda r:(r["score"],r["wilson_lcb_95"],r["trades"])) if pool else None

    use_meta=bool(
        selected
        and int(selected.get("trades",0) or 0)>=sample_floor
        and (
            float(selected.get("wilson_lcb_95",0.0))>=base_lcb+0.03
            or (
                bool(selected.get("target_met"))
                and float(selected.get("win_rate",0.0) or 0.0)>=base_wr+0.05
            )
        )
    )
    return {
        "mode":"META_VETO" if use_meta else "BYPASS",
        "threshold":None if not use_meta else float(selected["threshold"]),
        "selection":selected,
        "base":{
            "trades":base_trades,
            "win_rate":base_metrics.get("win_rate"),
            "expectancy_r":base_metrics.get("expectancy_r"),
            "profit_factor":base_metrics.get("profit_factor"),
            "wilson_lcb_95":base_lcb,
        },
        "target_win_rate":float(target_win_rate),
        "minimum_policy_trades":int(sample_floor),
        "minimum_policy_coverage":float(coverage_floor),
        "candidate_count":int(len(rows)),
    }
