@echo off
REM ------------------------------------------------------------------
REM One-click runner for Windows: create venv, install deps, run pipeline
REM Usage: double-click this file or run from a terminal: run_all.bat
REM ------------------------------------------------------------------

:: Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: Activate virtual environment
call venv\Scripts\activate
if errorlevel 1 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

echo Upgrading pip and installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install Python dependencies.
    pause
    exit /b 1
)

:: Ensure dataset exists
if not exist data\emails.csv (
    echo ERROR: data\emails.csv not found!
    echo Please place the dataset at data\emails.csv and try again.
    pause
    exit /b 1
)

:: Ensure output directories
if not exist data\processed (
    mkdir data\processed
)
if not exist reports (
    mkdir reports
)
if not exist models (
    mkdir models
)

REM ---- Phase 1: Data preparation ----
echo Running data preparation...
python src\data_prep.py --input data\emails.csv --output_dir data\processed --test_size 0.2 --random_state 42
if errorlevel 1 (
    echo Data preparation failed.
    pause
    exit /b 1
)

REM ---- Phase 2: Train baselines ----
echo Training baselines...
python src\train_baselines.py --train_csv data\processed\train.csv --report_out reports\baselines.json --cv_splits 5 --random_state 42
if errorlevel 1 (
    echo Baseline training failed.
    pause
    exit /b 1
)

REM ---- Phase 3: Train final model ----
echo Training final model...
python src\train_final.py --train_csv data\processed\train.csv --test_csv data\processed\test.csv --vocab_json data\processed\vocab.json --models_dir models --random_state 42
if errorlevel 1 (
    echo Final training failed.
    pause
    exit /b 1
)

REM ---- Run the web app ----
echo Starting web app (press Ctrl+C in this terminal to stop)...
python -m src.app

pause
