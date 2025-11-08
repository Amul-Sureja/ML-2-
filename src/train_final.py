import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train final model and evaluate on test set"
    )
    parser.add_argument(
        "--train_csv", type=Path, required=True, help="Path to processed train.csv"
    )
    parser.add_argument(
        "--test_csv", type=Path, required=True, help="Path to processed test.csv"
    )
    parser.add_argument(
        "--vocab_json", type=Path, required=True, help="Path to vocab.json"
    )
    parser.add_argument(
        "--models_dir",
        type=Path,
        required=True,
        help="Directory to save model artifacts",
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    with args.vocab_json.open("r") as f:
        vocab = json.load(f)["features"]

    # Ensure consistent column order
    X_train = train_df[vocab].values
    y_train = train_df["Prediction"].astype(int).values
    X_test = test_df[vocab].values
    y_test = test_df["Prediction"].astype(int).values

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=args.random_state,
    )
    model.fit(X_train, y_train)

    # Train-set metrics
    y_pred_train = model.predict(X_train)
    train_metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_train, y_pred_train)),
        "precision": float(precision_score(y_train, y_pred_train)),
        "recall": float(recall_score(y_train, y_pred_train)),
        "f1": float(f1_score(y_train, y_pred_train)),
    }

    # Test-set metrics
    y_pred = model.predict(X_test)
    y_prob = None
    try:
        y_prob = model.predict_proba(X_test)[:, 1].tolist()
    except Exception:
        y_prob = None

    test_metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
    }

    cm = confusion_matrix(y_test, y_pred).tolist()
    clf_report = classification_report(y_test, y_pred, output_dict=True)

    # Save model
    model_path = args.models_dir / "model.joblib"
    joblib.dump(model, model_path)

    # Save metadata
    metadata = {
        "model_type": "LogisticRegression",
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": args.random_state,
        "vocab_size": len(vocab),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "confusion_matrix": cm,
        "classification_report": clf_report,
    }
    with (args.models_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    # Echo a short summary
    print(
        json.dumps(
            {
                "model_path": str(model_path),
                "train": train_metrics,
                "test": test_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
