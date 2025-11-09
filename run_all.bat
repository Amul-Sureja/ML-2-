@echo off
setlocal
REM ------------------------------------------------------------------
REM One-click runner for Windows: create venv, install deps, run pipeline
REM Usage: double-click or run from a terminal: run_all.bat
REM ------------------------------------------------------------------

REM Use script directory as working base
cd /d "%~dp0"

:: Show which python will be used (debug help)
echo Locating Python...
where python || echo "python not found on PATH"
python --version || echo "Failed to run python --version"

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

:: Activate virtual environment (use explicit .bat)
echo Activating virtual environment...
call "%~dp0venv\Scripts\activate.bat"
if errorlevel 1 (
    echo Failed to activate virtual environment.
    echo If activation fails, try running this script from an elevated prompt or ensure venv exists.
    pause
    exit /b 1
)

echo Using python at:
where python

echo Upgrading pip and installing dependencies...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip.
    pause
    exit /b 1
)

REM Use python -m pip to avoid 'pip' not recognized
python -m pip install -r requirements.txt
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
endlocal



@REM @echo off
@REM REM ------------------------------------------------------------------
@REM REM One-click runner for Windows: create venv, install deps, run pipeline
@REM REM Usage: double-click this file or run from a terminal: run_all.bat
@REM REM ------------------------------------------------------------------

@REM :: Create virtual environment if it doesn't exist
@REM if not exist venv (
@REM     echo Creating virtual environment...
@REM     python -m venv venv
@REM     if errorlevel 1 (
@REM         echo Failed to create virtual environment.
@REM         pause
@REM         exit /b 1
@REM     )
@REM )

@REM :: Activate virtual environment
@REM call venv\Scripts\activate
@REM if errorlevel 1 (
@REM     echo Failed to activate virtual environment.
@REM     pause
@REM     exit /b 1
@REM )

@REM echo Upgrading pip and installing dependencies...
@REM python -m pip install --upgrade pip
@REM pip install -r requirements.txt
@REM if errorlevel 1 (
@REM     echo Failed to install Python dependencies.
@REM     pause
@REM     exit /b 1
@REM )

@REM :: Ensure dataset exists
@REM if not exist data\emails.csv (
@REM     echo ERROR: data\emails.csv not found!
@REM     echo Please place the dataset at data\emails.csv and try again.
@REM     pause
@REM     exit /b 1
@REM )

@REM :: Ensure output directories
@REM if not exist data\processed (
@REM     mkdir data\processed
@REM )
@REM if not exist reports (
@REM     mkdir reports
@REM )
@REM if not exist models (
@REM     mkdir models
@REM )

@REM REM ---- Phase 1: Data preparation ----
@REM echo Running data preparation...
@REM python src\data_prep.py --input data\emails.csv --output_dir data\processed --test_size 0.2 --random_state 42
@REM if errorlevel 1 (
@REM     echo Data preparation failed.
@REM     pause
@REM     exit /b 1
@REM )

@REM REM ---- Phase 2: Train baselines ----
@REM echo Training baselines...
@REM python src\train_baselines.py --train_csv data\processed\train.csv --report_out reports\baselines.json --cv_splits 5 --random_state 42
@REM if errorlevel 1 (
@REM     echo Baseline training failed.
@REM     pause
@REM     exit /b 1
@REM )

@REM REM ---- Phase 3: Train final model ----
@REM echo Training final model...
@REM python src\train_final.py --train_csv data\processed\train.csv --test_csv data\processed\test.csv --vocab_json data\processed\vocab.json --models_dir models --random_state 42
@REM if errorlevel 1 (
@REM     echo Final training failed.
@REM     pause
@REM     exit /b 1
@REM )

@REM REM ---- Run the web app ----
@REM echo Starting web app (press Ctrl+C in this terminal to stop)...
@REM python -m src.app

@REM pause