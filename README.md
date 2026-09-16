# PMLDL Assignment 1 - Automated MLOps Pipeline

## Project overview

This university project predicts Titanic passenger survival and implements an automated MLOps pipeline covering data preparation, model training, experiment tracking, deployment, and scheduled execution.

## Pipeline

### Data Engineering

- Loads the raw Titanic dataset.
- Handles missing values.
- Handles `Age` and `Fare` outliers using IQR clipping.
- Splits the processed data into training and test sets.

### Model Engineering

- Applies preprocessing with scikit-learn.
- Trains a Logistic Regression model.
- Evaluates accuracy, precision, recall, and F1 score.
- Logs experiments and metrics with MLflow.
- Saves the trained model with joblib.

### Deployment

- Serves predictions through a FastAPI API.
- Provides a Streamlit application.
- Runs the API and application in separate Docker containers.
- Orchestrates the containers with Docker Compose.

## Project structure

```text
.
|-- code/
|   |-- datasets/preprocess.py
|   |-- models/train.py
|   `-- deployment/
|       |-- api/
|       |   |-- main.py
|       |   `-- Dockerfile
|       |-- app/
|       |   |-- app.py
|       |   `-- Dockerfile
|       `-- docker-compose.yml
|-- data/
|   |-- raw/train.csv
|   `-- processed/
|       |-- train.csv
|       `-- test.csv
|-- metrics/metrics.json
|-- models/model.joblib
|-- dvc.yaml
|-- dvc.lock
|-- requirements.txt
`-- run_pipeline.ps1
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the complete pipeline

```powershell
dvc repro --force
```

## Run Docker manually

```powershell
docker compose -f code/deployment/docker-compose.yml up --build
```

## Service URLs

- FastAPI: [http://localhost:8000](http://localhost:8000)
- Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)
- Streamlit: [http://localhost:8501](http://localhost:8501)
- MLflow: [http://localhost:5000](http://localhost:5000)

## Automation

`run_pipeline.ps1` is executed every five minutes using Windows Task Scheduler. The script runs the DVC pipeline with the DVC executable from the project's local virtual environment.

## Model inputs

- `Pclass`
- `Sex`
- `Age`
- `SibSp`
- `Parch`
- `Fare`
- `Embarked`
