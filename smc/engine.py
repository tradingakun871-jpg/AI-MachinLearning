import pandas as pd
from smc.structure import market_structure
from smc.imbalances import fair_value_gaps, liquidity_sweeps, premium_discount
from smc.order_blocks import order_blocks
from smc.session import session_features
from smc.price_action import add_price_action_features
from smc.orderflow import add_orderflow_features


def add_smc_features(df: pd.DataFrame):
    d = market_structure(df)
    d = fair_value_gaps(d)
    d = liquidity_sweeps(d)
    d = premium_discount(d)
    d = order_blocks(d)
    d = session_features(d)

    # Context layers are causal and are shared by training and live inference.
    # Price Action is always computed. Order Flow uses real delta/volume when
    # available and otherwise marks its OHLCV/tick-volume estimate as a proxy.
    d = add_price_action_features(d)
    d = add_orderflow_features(d)

    d["mss"] = d["choch"]
    d["liquidity_sweep"] = (d["liquidity_sweep_direction"] != 0).astype(int)
    d["order_block"] = (d["ob_direction"] != 0).astype(int)
    d["fvg"] = (d["fvg_direction"] != 0).astype(int)
    return d
