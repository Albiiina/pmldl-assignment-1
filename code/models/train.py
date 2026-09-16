import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "Survived"
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    train_path = project_root / "data" / "processed" / "train.csv"
    test_path = project_root / "data" / "processed" / "test.csv"
    metrics_directory = project_root / "metrics"
    models_directory = project_root / "models"
    mlflow_database = project_root / "mlflow.db"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data[FEATURES]
    y_train = train_data[TARGET]
    X_test = test_data[FEATURES]
    y_test = test_data[TARGET]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
    }

    print("Model evaluation metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    metrics_directory.mkdir(parents=True, exist_ok=True)
    models_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_directory / "metrics.json"
    model_path = models_directory / "model.joblib"

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=4)

    joblib.dump(model, model_path)

    mlflow.set_tracking_uri(f"sqlite:///{mlflow_database.resolve().as_posix()}")
    mlflow.set_experiment("Titanic Survival")
    with mlflow.start_run():
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metrics(metrics)

    print(f"Metrics saved to {metrics_path}")
    print(f"Model saved to {model_path}")
    print(f"MLflow run logged locally in {mlflow_database}")


if __name__ == "__main__":
    main()
