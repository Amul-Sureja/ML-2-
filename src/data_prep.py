import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare dataset: split and save artifacts"
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to emails.csv")
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write processed artifacts",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Test split size fraction"
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    # Validate expected columns
    if "Prediction" not in df.columns:
        raise ValueError("'Prediction' column not found in dataset")

    # Drop identifier column if present
    drop_cols = [c for c in ["Email No."] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Separate features and target
    feature_columns = [c for c in df.columns if c != "Prediction"]
    X = df[feature_columns]
    y = df["Prediction"].astype(int)

    # Basic validation
    if set(y.unique()) - {0, 1}:
        raise ValueError("'Prediction' must be binary (0/1)")

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Save split CSVs with label
    train_out = pd.concat(
        [X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1
    )
    test_out = pd.concat(
        [X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1
    )

    train_out.to_csv(output_dir / "train.csv", index=False)
    test_out.to_csv(output_dir / "test.csv", index=False)

    # Save vocabulary (feature order)
    with (output_dir / "vocab.json").open("w") as f:
        json.dump({"features": feature_columns}, f, indent=2)

    # Summary metadata
    summary = {
        "n_rows": int(df.shape[0]),
        "n_features": int(len(feature_columns)),
        "class_distribution": {
            str(int(k)): int(v) for k, v in y.value_counts().to_dict().items()
        },
        "test_size": args.test_size,
        "random_state": args.random_state,
        "dropped_columns": drop_cols,
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # Console summary
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
