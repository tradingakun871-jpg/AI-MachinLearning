import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def trading_metrics(y, p, threshold=0.60, rr=3.0):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float); take=p>=threshold
    yt=y[take]; pt=p[take]
    if len(yt)==0: return {"trades":0}
    r=np.where(yt==1,rr,-1.0); equity=np.cumsum(r); peak=np.maximum.accumulate(np.r_[0.0,equity])[1:]; dd=equity-peak
    gross_win=float(r[r>0].sum()); gross_loss=float(-r[r<0].sum())
    return {"trades":int(len(r)),"win_rate":float(yt.mean()),"net_r":float(r.sum()),"expectancy_r":float(r.mean()),"profit_factor":gross_win/gross_loss if gross_loss>0 else None,"max_drawdown_r":float(dd.min()) if len(dd) else 0.0,"avg_probability":float(pt.mean())}


def probability_metrics(y,p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    return {"brier":float(brier_score_loss(y,p)),"logloss":float(log_loss(y,np.c_[1-p,p],labels=[0,1])),"auc":float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None}


def grouped_performance(frame, group_col, threshold=0.60, rr=3.0):
    out={}
    for key,g in frame.groupby(group_col,dropna=False): out[str(key)]=trading_metrics(g.label,g.probability,threshold,rr)
    return out
