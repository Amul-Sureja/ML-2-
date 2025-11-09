#!/usr/bin/env bash
# ------------------------------------------------------------------
# One-click runner for macOS / Linux: create venv, install deps, run pipeline
# Usage: double-click this file or run from terminal: ./run_all.sh
# ------------------------------------------------------------------

set -euo pipefail
IFS=$'\n\t'

# Make sure we're in the script directory
cd "$(dirname "$0")"

# Prefer python3 when creating venv
PYTHON_CMD="${PYTHON_CMD:-python3}"

# Check python availability
if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_CMD not found. Install Python 3 or set PYTHON_CMD to point to your python binary."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment with $PYTHON_CMD..."
    "$PYTHON_CMD" -m venv venv
fi

VENV_PY="./venv/bin/python"
VENV_PIP="./venv/bin/python -m pip"

# Activate virtualenv for subshells (mostly useful for interactive sessions)
# shellcheck disable=SC1091
if [ -f "venv/bin/activate" ]; then
    # shell will source activation for interactive shells; keep it optional
    # but rely on VENV_PY and VENV_PIP for commands below to be deterministic
    : # no-op
fi

echo "Using virtualenv python at: $VENV_PY"
"$VENV_PY" --version

echo "Upgrading pip and installing dependencies..."
# use the venv's python -m pip to avoid relying on a global pip executable
"$VENV_PIP" install --upgrade pip
"$VENV_PIP" install -r requirements.txt

# Ensure dataset exists
if [ ! -f "data/emails.csv" ]; then
    echo "ERROR: data/emails.csv not found!"
    echo "Please place the dataset at data/emails.csv and try again."
    exit 1
fi

# Ensure output directories
mkdir -p data/processed
mkdir -p reports
mkdir -p models

# ---- Phase 1: Data preparation ----
echo "Running data preparation..."
"$VENV_PY" src/data_prep.py --input data/emails.csv --output_dir data/processed --test_size 0.2 --random_state 42

# ---- Phase 2: Train baselines ----
echo "Training baselines..."
"$VENV_PY" src/train_baselines.py --train_csv data/processed/train.csv --report_out reports/baselines.json --cv_splits 5 --random_state 42

# ---- Phase 3: Train final model ----
echo "Training final model..."
"$VENV_PY" src/train_final.py --train_csv data/processed/train.csv --test_csv data/processed/test.csv --vocab_json data/processed/vocab.json --models_dir models --random_state 42

# ---- Run the web app ----
echo "Starting web app (press Ctrl+C in this terminal to stop)..."
"$VENV_PY" -m src.app

# Keep terminal open (only relevant if double-clicked)
echo
read -n 1 -r -p "Press any key to continue..."
echo
