import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline models with CV on train split")
    parser.add_argument("--train_csv", type=Path, required=True, help="Path to processed train.csv")
    parser.add_argument("--report_out", type=Path, required=True, help="Path to write baselines report JSON")
    parser.add_argument("--cv_splits", type=int, default=5, help="Number of CV splits")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def evaluate_models(X: np.ndarray, y: np.ndarray, cv_splits: int, random_state: int) -> Tuple[Dict, str]:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    models: Dict[str, object] = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
        "linearsvc_calibrated": CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=5),
        "multinomial_nb": MultinomialNB(),
    }

    report: Dict[str, Dict[str, float]] = {}
    best_model_name = None
    best_mean_accuracy = -1.0

    for name, model in models.items():
        acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=None)
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=None)
        report[name] = {
            "mean_accuracy": float(np.mean(acc_scores)),
            "std_accuracy": float(np.std(acc_scores)),
            "mean_f1": float(np.mean(f1_scores)),
            "std_f1": float(np.std(f1_scores)),
        }

        if report[name]["mean_accuracy"] > best_mean_accuracy:
            best_mean_accuracy = report[name]["mean_accuracy"]
            best_model_name = name

    return report, best_model_name or ""


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.train_csv)

    if "Prediction" not in df.columns:
        raise ValueError("'Prediction' column not found in train.csv")

    feature_columns: List[str] = [c for c in df.columns if c != "Prediction"]
    X = df[feature_columns].values
    y = df["Prediction"].astype(int).values

    report, best_name = evaluate_models(X, y, cv_splits=args.cv_splits, random_state=args.random_state)

    out = {
        "cv_splits": args.cv_splits,
        "random_state": args.random_state,
        "n_rows_train": int(df.shape[0]),
        "n_features": int(len(feature_columns)),
        "results": report,
        "selected_by": "mean_accuracy",
        "best_model": best_name,
    }

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with args.report_out.open("w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


