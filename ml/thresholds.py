import numpy as np
from ml.performance import trading_metrics


def threshold_sweep(y,p,rr=3.0,start=.50,end=.85,step=.01,min_trades=30):
    """Research helper. Select thresholds on validation data only, never on final test data."""
    rows=[]
    for t in np.arange(start,end+1e-9,step):
        m=trading_metrics(y,p,float(t),rr)
        if m.get("trades",0)>=min_trades: rows.append({"threshold":round(float(t),3),**m})
    return rows
