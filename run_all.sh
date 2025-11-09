#!/bin/bash
# ------------------------------------------------------------------
# One-click runner for macOS: create venv, install deps, run pipeline
# Usage: double-click this file or run from terminal: ./run_all.sh
# ------------------------------------------------------------------

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment."
        exit 1
    fi
fi

# Activate virtual environment
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment."
    exit 1
fi

echo "Upgrading pip and installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install Python dependencies."
    exit 1
fi

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
python src/data_prep.py --input data/emails.csv --output_dir data/processed --test_size 0.2 --random_state 42
if [ $? -ne 0 ]; then
    echo "Data preparation failed."
    exit 1
fi

# ---- Phase 2: Train baselines ----
echo "Training baselines..."
python src/train_baselines.py --train_csv data/processed/train.csv --report_out reports/baselines.json --cv_splits 5 --random_state 42
if [ $? -ne 0 ]; then
    echo "Baseline training failed."
    exit 1
fi

# ---- Phase 3: Train final model ----
echo "Training final model..."
python src/train_final.py --train_csv data/processed/train.csv --test_csv data/processed/test.csv --vocab_json data/processed/vocab.json --models_dir models --random_state 42
if [ $? -ne 0 ]; then
    echo "Final training failed."
    exit 1
fi

# ---- Run the web app ----
echo "Starting web app (press Ctrl+C in this terminal to stop)..."
python -m src.app

# Keep terminal open (only relevant if double-clicked)
echo "Press any key to continue..."
read -n 1