from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET = "Survived"
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


def clip_outliers(data: pd.DataFrame, column: str) -> None:
    """Clip values in a numeric column to its IQR bounds."""
    first_quartile = data[column].quantile(0.25)
    third_quartile = data[column].quantile(0.75)
    iqr = third_quartile - first_quartile

    lower_bound = first_quartile - 1.5 * iqr
    upper_bound = third_quartile + 1.5 * iqr
    data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data" / "raw" / "train.csv"
    output_directory = project_root / "data" / "processed"

    data = pd.read_csv(input_path)
    print(f"Original dataset shape: {data.shape}")

    selected_columns = [TARGET] + FEATURES
    data = data[selected_columns].copy()

    missing_values = data.isna().sum()
    print("Missing values before cleaning:")
    print(missing_values.to_string())
    print(f"Total missing values: {missing_values.sum()}")

    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

    clip_outliers(data, "Age")
    clip_outliers(data, "Fare")

    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data[TARGET],
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    train_path = output_directory / "train.csv"
    test_path = output_directory / "test.csv"

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"Train dataset shape: {train_data.shape}")
    print(f"Test dataset shape: {test_data.shape}")
    print(f"Files saved to {train_path} and {test_path}")


if __name__ == "__main__":
    main()
