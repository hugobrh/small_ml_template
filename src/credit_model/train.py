from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train():
    # 1. Load Data
    data = fetch_openml(data_id=31, as_frame=True, parser="auto")
    X, y = data.data, data.target
    y = (y == "good").astype(int)  # Binary classification

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Preprocessing
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["category", "object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # 3. Pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    with mlflow.start_run():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        mlflow.log_metric("accuracy", score)
        mlflow.sklearn.log_model(model, "model")

        # Save locally for API
        Path("models").mkdir(exist_ok=True)
        import joblib
        joblib.dump(model, "models/lr.joblib")
        print(f"LR model trained with accuracy: {score}")

    # # 4. Print features and types for API mapping
    # feature_names = X_train.columns
    # feature_dtypes = X_train.dtypes
    # print("Feature names and types:")
    # for name, dtype in zip(feature_names, feature_dtypes):
    #     print(f"{name}: {dtype}")

if __name__ == "__main__":
    train()
