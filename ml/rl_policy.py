import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from ml.performance import trading_metrics


class OfflineQPolicy:
    """Research-only fitted Q-learning selector for TRADE vs SKIP.

    The historical label supplies a counterfactual reward for taking the trade:
    +RR when TP wins and -1R when SL wins. SKIP reward is 0. The next candidate
    is used as the next state with a small discount factor. This is an offline
    policy filter, not a live broker execution engine.
    """

    def __init__(self,gamma=0.10,iterations=4,max_depth=4,learning_rate=0.05):
        self.gamma=float(gamma)
        self.iterations=int(iterations)
        self.model=HistGradientBoostingRegressor(
            max_depth=int(max_depth),
            learning_rate=float(learning_rate),
            max_iter=180,
            l2_regularization=1.0,
            random_state=42,
        )
        self.margin=0.0
        self.margin_selection=None

    @staticmethod
    def _reward(y,rr=3.0):
        y=np.asarray(y,dtype=int)
        return np.where(y==1,float(rr),-1.0)

    @staticmethod
    def _expand(states):
        x=np.asarray(states,dtype=float)
        n=len(x)
        skip=np.c_[x,np.zeros(n)]
        trade=np.c_[x,np.ones(n)]
        return np.vstack([skip,trade])

    def fit(self,states,y,rr=3.0):
        x=np.asarray(states,dtype=float)
        y=np.asarray(y,dtype=int)
        if len(x)<50:
            raise ValueError("insufficient rows for offline Q policy")

        rewards=self._reward(y,rr)
        expanded=self._expand(x)
        immediate=np.r_[np.zeros(len(x)),rewards]

        targets=immediate.copy()
        for _ in range(max(1,self.iterations)):
            self.model.fit(expanded,targets)

            q_skip=self.model.predict(np.c_[x,np.zeros(len(x))])
            q_trade=self.model.predict(np.c_[x,np.ones(len(x))])
            next_value=np.maximum(q_skip,q_trade)
            future=np.r_[next_value[1:],0.0]

            targets=np.r_[
                self.gamma*future,
                rewards+self.gamma*future,
            ]
        self.model.fit(expanded,targets)
        return self

    def q_values(self,states):
        x=np.asarray(states,dtype=float)
        q_skip=self.model.predict(np.c_[x,np.zeros(len(x))])
        q_trade=self.model.predict(np.c_[x,np.ones(len(x))])
        return q_skip,q_trade

    def advantage(self,states):
        q_skip,q_trade=self.q_values(states)
        return np.asarray(q_trade-q_skip,dtype=float)

    def calibrate_margin(self,states,y,rr=3.0,min_trades=None,min_coverage=0.03):
        adv=self.advantage(states)
        y=np.asarray(y,dtype=int)
        if min_trades is None:
            min_trades=max(10,int(len(y)*0.03))

        qs=np.quantile(adv,[.25,.35,.45,.55,.65,.72,.78,.84,.89,.93,.96])
        candidates=sorted({round(float(x),4) for x in np.r_[0.0,qs] if np.isfinite(x)})
        rows=[]

        for margin in candidates:
            mask=adv>=margin
            trades=int(mask.sum())
            coverage=float(trades/len(y)) if len(y) else 0.0
            if trades<min_trades or coverage<min_coverage:
                continue
            metrics=trading_metrics(y[mask],adv[mask],threshold=-1e9,rr=rr)
            expectancy=float(metrics.get("expectancy_r",-999) or -999)
            pf=metrics.get("profit_factor")
            dd=abs(float(metrics.get("max_drawdown_r",0.0) or 0.0))
            score=expectancy*np.sqrt(trades)-0.01*dd
            rows.append({
                "margin":float(margin),
                "coverage":coverage,
                "score":float(score),
                **metrics,
            })

        viable=[
            row for row in rows
            if float(row.get("expectancy_r",-999))>0
            and row.get("profit_factor") is not None
            and float(row.get("profit_factor",0))>=1.05
        ]
        selected=max(viable,key=lambda x:(x["score"],x["trades"])) if viable else None
        if selected is not None:
            self.margin=float(selected["margin"])
        else:
            self.margin=float("inf")
        self.margin_selection=selected
        return {
            "mode":"OFFLINE_Q" if selected else "NONE",
            "margin":None if selected is None else float(selected["margin"]),
            "selection":selected,
            "candidate_count":len(rows),
        }

    def select(self,states):
        adv=self.advantage(states)
        return adv>=self.margin,adv


def q_diagnostics(policy,states,y,rr=3.0):
    mask,adv=policy.select(states)
    y=np.asarray(y,dtype=int)
    metrics=(
        trading_metrics(y[mask],adv[mask],threshold=-1e9,rr=rr)
        if np.any(mask) else {"trades":0}
    )
    return {
        "rows":int(len(y)),
        "selected_rows":int(mask.sum()),
        "coverage":float(mask.mean()) if len(mask) else 0.0,
        "advantage_mean":float(np.mean(adv)) if len(adv) else None,
        "advantage_p90":float(np.quantile(adv,.90)) if len(adv) else None,
        "trading":metrics,
    }
