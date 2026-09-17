import numpy as np
import pandas as pd


def fair_value_gaps(df: pd.DataFrame):
    d=df.copy(); d["bull_fvg"]=(d.low>d.high.shift(2)).astype(int); d["bear_fvg"]=(d.high<d.low.shift(2)).astype(int)
    d["fvg_direction"]=np.where(d.bull_fvg,1,np.where(d.bear_fvg,-1,0)); d["fvg_size"] = np.where(d.bull_fvg,d.low-d.high.shift(2),np.where(d.bear_fvg,d.low.shift(2)-d.high,0.0))
    return d


def liquidity_sweeps(df: pd.DataFrame, lookback=20):
    d=df.copy(); prior_hi=d.high.shift(1).rolling(lookback).max(); prior_lo=d.low.shift(1).rolling(lookback).min()
    d["sweep_high"]=((d.high>prior_hi)&(d.close<prior_hi)).astype(int); d["sweep_low"]=((d.low<prior_lo)&(d.close>prior_lo)).astype(int)
    d["liquidity_sweep_direction"]=np.where(d.sweep_low,1,np.where(d.sweep_high,-1,0)); return d


def premium_discount(df: pd.DataFrame, lookback=50):
    d=df.copy(); hi=d.high.shift(1).rolling(lookback).max(); lo=d.low.shift(1).rolling(lookback).min(); eq=(hi+lo)/2
    rng=(hi-lo).replace(0,np.nan); d["dealing_range_position"]=(d.close-lo)/rng; d["discount"]=(d.close<eq).astype(int); d["premium"]=(d.close>eq).astype(int); return d
