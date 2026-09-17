import numpy as np
import pandas as pd


def session_features(df: pd.DataFrame):
    """UTC session flags; keep timezone handling explicit in upstream ingestion."""
    d=df.copy(); ts=pd.to_datetime(d.timestamp,utc=True); h=ts.dt.hour
    d["session_asia"]=((h>=0)&(h<8)).astype(int); d["session_london"]=((h>=7)&(h<16)).astype(int); d["session_newyork"]=((h>=12)&(h<21)).astype(int); d["session_overlap"]=((h>=12)&(h<16)).astype(int)
    return d
