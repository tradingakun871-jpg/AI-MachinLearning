import argparse
import json
import joblib
import pandas as pd
from pathlib import Path
from ml.features import FEATURE_COLUMNS
from ml.regime import RegimeClassifier
from ml.model import ProbabilityEnsemble
from ml.walk_forward import walk_forward


def main(csv_path, output_dir):
    df = pd.read_csv(csv_path).sort_values("timestamp").dropna(subset=["label"]).reset_index(drop=True)
    regime = RegimeClassifier().fit(df.iloc[:max(1, int(len(df)*0.7))])
    df["regime"] = regime.predict(df)
    features = FEATURE_COLUMNS + ["regime"]
    metrics = walk_forward(df, features)

    split = int(len(df)*0.8)
    cal_size = max(100, int(len(df)*0.1))
    train = df.iloc[:max(1, split-cal_size)]
    cal = df.iloc[max(1, split-cal_size):split]
    model = ProbabilityEnsemble().fit(train[features], train["label"]).calibrate(cal[features], cal["label"])

    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "regime": regime, "features": features}, out/"model.joblib")
    (out/"walk_forward.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"rows": len(df), "features": features, "walk_forward_folds": len(metrics), "output": str(out)}, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--output", default="artifacts/model-v0.2")
    a = p.parse_args(); main(a.csv, a.output)
