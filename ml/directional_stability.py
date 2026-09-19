import numpy as np

from ml.meta_precision import wilson_lower_bound


def _median(values):
    vals=[]
    for value in values:
        if value is None:
            continue
        try:
            number=float(value)
        except (TypeError,ValueError):
            continue
        if np.isfinite(number):
            vals.append(number)
    return float(np.median(vals)) if vals else None


def evaluate_directional_stability(
    temporal_stability,
    rr=2.0,
    minimum_usable_folds=3,
    minimum_positive_folds=2,
    minimum_win_rate=0.36,
    minimum_profit_factor=1.05,
    minimum_trades_per_fold=15,
):
    """Evaluate BUY and SELL separately from pre-test walk-forward folds.

    The gate uses development-only temporal folds. It never looks at the final
    holdout when deciding whether a direction may be enabled for shadow use.
    """
    folds=(temporal_stability or {}).get("folds") or []
    break_even=1.0/(1.0+float(rr))
    result={}

    for side in ("BUY","SELL"):
        side_rows=[]
        for fold in folds:
            side_info=((fold.get("sides") or {}).get(side) or {})
            trade=side_info.get("trading") or {}
            trades=int(trade.get("trades",0) or 0)
            row={
                "fold":fold.get("fold"),
                "policy_mode":side_info.get("policy_mode"),
                "trades":trades,
                "win_rate":trade.get("win_rate"),
                "expectancy_r":trade.get("expectancy_r"),
                "profit_factor":trade.get("profit_factor"),
                "max_drawdown_r":trade.get("max_drawdown_r"),
            }
            side_rows.append(row)

        usable=[row for row in side_rows if int(row.get("trades",0) or 0)>=int(minimum_trades_per_fold)]
        positive=[
            row for row in usable
            if float(row.get("expectancy_r",-999) or -999)>0
            and float(row.get("win_rate",0.0) or 0.0)>=float(minimum_win_rate)
            and row.get("profit_factor") is not None
            and float(row.get("profit_factor",0.0))>=float(minimum_profit_factor)
        ]

        pooled_trades=sum(int(row.get("trades",0) or 0) for row in usable)
        pooled_wins=sum(
            int(round(float(row.get("win_rate",0.0) or 0.0)*int(row.get("trades",0) or 0)))
            for row in usable
        )
        pooled_lcb=wilson_lower_bound(pooled_wins,pooled_trades) if pooled_trades else 0.0
        median_wr=_median([row.get("win_rate") for row in usable])
        median_exp=_median([row.get("expectancy_r") for row in usable])
        median_pf=_median([row.get("profit_factor") for row in usable])

        passed=bool(
            len(usable)>=int(minimum_usable_folds)
            and len(positive)>=int(minimum_positive_folds)
            and median_wr is not None and median_wr>=float(minimum_win_rate)
            and median_exp is not None and median_exp>0
            and median_pf is not None and median_pf>=float(minimum_profit_factor)
            and pooled_lcb>=break_even
        )
        result[side]={
            "passed":passed,
            "usable_folds":int(len(usable)),
            "positive_folds":int(len(positive)),
            "median_win_rate":median_wr,
            "median_expectancy_r":median_exp,
            "median_profit_factor":median_pf,
            "pooled_trades":int(pooled_trades),
            "pooled_wins":int(pooled_wins),
            "pooled_wilson_lcb_95":float(pooled_lcb),
            "folds":side_rows,
            "requirements":{
                "minimum_usable_folds":int(minimum_usable_folds),
                "minimum_positive_folds":int(minimum_positive_folds),
                "minimum_median_win_rate":float(minimum_win_rate),
                "minimum_median_expectancy_r":0.0,
                "minimum_median_profit_factor":float(minimum_profit_factor),
                "minimum_pooled_wilson_lcb_95":float(break_even),
                "minimum_trades_per_usable_fold":int(minimum_trades_per_fold),
            },
        }

    result["any_direction_passed"]=bool(
        result.get("BUY",{}).get("passed") or result.get("SELL",{}).get("passed")
    )
    return result
