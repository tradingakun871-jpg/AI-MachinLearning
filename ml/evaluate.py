import argparse,json,joblib
from pathlib import Path
import pandas as pd
from ml.performance import trading_metrics,probability_metrics,grouped_performance


def evaluate(csv_path,artifact_path,output,threshold=0.60,rr=3.0):
    df=pd.read_csv(csv_path).sort_values("timestamp").dropna(subset=["label"]).reset_index(drop=True); bundle=joblib.load(artifact_path)
    df["regime"]=bundle["regime"].predict(df); features=bundle["features"]; df["probability"]=bundle["model"].predict_proba(df[features])
    report={"all_probability":probability_metrics(df.label,df.probability),"all_trading":trading_metrics(df.label,df.probability,threshold,rr),"by_regime":grouped_performance(df,"regime",threshold,rr)}
    for col in ["session_asia","session_london","session_newyork","session_overlap"]:
        if col in df: report[f"by_{col}"]=grouped_performance(df,col,threshold,rr)
    Path(output).parent.mkdir(parents=True,exist_ok=True); Path(output).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("csv"); p.add_argument("artifact"); p.add_argument("--output",default="artifacts/performance.json"); p.add_argument("--threshold",type=float,default=.60); p.add_argument("--rr",type=float,default=3.0); a=p.parse_args(); evaluate(a.csv,a.artifact,a.output,a.threshold,a.rr)
