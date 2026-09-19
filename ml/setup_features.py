import numpy as np
import pandas as pd


def signal_direction(frame: pd.DataFrame) -> pd.Series:
    """Directional intent from causal SMC structure events only."""
    d=frame
    direction=np.zeros(len(d),dtype=int)

    for col in ("mss","bos","liquidity_sweep_direction"):
        if col not in d:
            continue
        values=np.sign(pd.to_numeric(d[col],errors="coerce").fillna(0).to_numpy()).astype(int)
        empty=direction==0
        direction=np.where(empty & (values!=0),values,direction)

    if "structure_bias" in d:
        bias=np.sign(pd.to_numeric(d["structure_bias"],errors="coerce").fillna(0).to_numpy()).astype(int)
        direction=np.where(direction==0,bias,direction)

    return pd.Series(direction,index=d.index,dtype=int)


def add_setup_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add causal SMC + price-action + order-flow setup context.

    SMC remains the event generator. Price Action and Order Flow are context
    features first; they do not become hard entry vetoes until walk-forward
    validation proves that doing so improves net expectancy and stability.
    """
    d=frame.copy()
    direction=signal_direction(d)
    d["signal_direction"]=direction

    def aligned(col):
        values=np.sign(pd.to_numeric(d.get(col,0),errors="coerce").fillna(0)).astype(int)
        return ((values!=0) & (values==direction)).astype(int)

    d["signal_mss"]=aligned("mss")
    d["signal_bos"]=aligned("bos")
    d["signal_sweep"]=aligned("liquidity_sweep_direction")
    d["signal_ob"]=aligned("ob_direction")
    d["signal_fvg"]=aligned("fvg_direction")

    d["signal_pa"]=aligned("pa_direction")
    d["signal_pa_breakout"]=aligned("pa_breakout_direction")
    d["signal_pa_rejection"]=aligned("pa_rejection_direction")
    d["signal_orderflow"]=aligned("orderflow_direction")
    if "orderflow_available" in d:
        d["signal_orderflow"]=(
            d["signal_orderflow"]
            * pd.to_numeric(d["orderflow_available"],errors="coerce").fillna(0).astype(int)
        )

    d["primary_structure_count"]=d[["signal_mss","signal_bos","signal_sweep"]].sum(axis=1)
    d["imbalance_count"]=d[["signal_ob","signal_fvg"]].sum(axis=1)
    d["smc_confluence"]=d[
        ["signal_mss","signal_bos","signal_sweep","signal_ob","signal_fvg"]
    ].sum(axis=1)
    d["price_action_confluence"]=d[
        ["signal_pa","signal_pa_breakout","signal_pa_rejection"]
    ].sum(axis=1)
    d["orderflow_confluence"]=d["signal_orderflow"]

    of_quality=pd.to_numeric(d.get("orderflow_source_quality",0.0),errors="coerce").fillna(0.0)
    d["context_confluence"]=(
        d["smc_confluence"].astype(float)
        +0.50*d["price_action_confluence"].astype(float)
        +0.50*d["orderflow_confluence"].astype(float)*of_quality
    )

    for tf in ("trend_m5","trend_m15"):
        if tf not in d:
            d[tf]=0
    d["htf_alignment"]=(
        (np.sign(d["trend_m5"].fillna(0)).astype(int)==direction).astype(int)
        +(np.sign(d["trend_m15"].fillna(0)).astype(int)==direction).astype(int)
    )
    d.loc[direction.eq(0),"htf_alignment"]=0

    # Distance to known structure at the signal close. No future price is used.
    atr=pd.to_numeric(d.get("atr",np.nan),errors="coerce").replace(0,np.nan)
    close=pd.to_numeric(d["close"],errors="coerce")
    swing_hi=pd.to_numeric(d.get("swing_high",np.nan),errors="coerce")
    swing_lo=pd.to_numeric(d.get("swing_low",np.nan),errors="coerce")
    d["distance_to_swing_atr"]=np.where(
        direction>0,(close-swing_lo).abs()/atr,
        np.where(direction<0,(swing_hi-close).abs()/atr,np.nan)
    )

    # Candidate still requires a real SMC structural event. PA/Order Flow are
    # learned confirmations, not standalone signal generators.
    d["candidate"]=((direction!=0) & (d["primary_structure_count"]>0)).astype(int)
    return d


def session_name(row) -> str:
    if int(row.get("session_overlap",0) or 0):
        return "overlap"
    if int(row.get("session_london",0) or 0):
        return "london"
    if int(row.get("session_newyork",0) or 0):
        return "newyork"
    if int(row.get("session_asia",0) or 0):
        return "asia"
    return "off"


def add_session_name(frame: pd.DataFrame) -> pd.DataFrame:
    d=frame.copy()
    d["session_name"]=[session_name(row) for _,row in d.iterrows()]
    return d


def structural_stop_distance(row, entry, min_atr=0.75, max_atr=2.5, buffer_atr=0.10):
    """Protect behind the nearest causal swing/OB level with ATR bounds."""
    atr=float(row.get("atr",0) or 0)
    if not np.isfinite(atr) or atr<=0:
        return None

    direction=int(row.get("signal_direction",0) or 0)
    entry=float(entry)
    levels=[]

    if direction>0:
        for key in ("swing_low","ob_low"):
            value=row.get(key)
            if value is not None and pd.notna(value) and float(value)<entry:
                levels.append(float(value))
        raw=entry-max(levels) if levels else atr
    elif direction<0:
        for key in ("swing_high","ob_high"):
            value=row.get(key)
            if value is not None and pd.notna(value) and float(value)>entry:
                levels.append(float(value))
        raw=min(levels)-entry if levels else atr
    else:
        return None

    raw=max(raw+buffer_atr*atr,min_atr*atr)
    return min(raw,max_atr*atr)
