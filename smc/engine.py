import pandas as pd
from smc.structure import market_structure
from smc.imbalances import fair_value_gaps, liquidity_sweeps, premium_discount
from smc.order_blocks import order_blocks
from smc.session import session_features


def add_smc_features(df: pd.DataFrame):
    d=market_structure(df); d=fair_value_gaps(d); d=liquidity_sweeps(d); d=premium_discount(d); d=order_blocks(d); d=session_features(d)
    d["mss"]=d["choch"]
    d["liquidity_sweep"]=(d["liquidity_sweep_direction"]!=0).astype(int)
    d["order_block"]=(d["ob_direction"]!=0).astype(int)
    d["fvg"]=(d["fvg_direction"]!=0).astype(int)
    return d
