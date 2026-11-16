"""
Train script for MLflow Projects — run via `mlflow run 06-projects/`.

Accepts parameters from the MLproject file, trains a Ridge regression model,
logs everything to MLflow, and registers the model in the registry.
"""

import argparse

import mlflow
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main(n_samples: int, noise_level: float, test_size: float, alpha: float) -> None:
    np.random.seed(42)
    X = np.random.rand(n_samples, 3) * 10
    y = 2.5 * X[:, 0] + 1.5 * X[:, 1] - 1.0 * X[:, 2] + 3 + np.random.randn(n_samples) * noise_level

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    with mlflow.start_run():
        mlflow.log_params({
            "n_samples":   n_samples,
            "noise_level": noise_level,
            "test_size":   test_size,
            "alpha":       alpha,
            "model_type":  "Ridge",
        })

        pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=alpha))])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mse  = mean_squared_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = mse ** 0.5
        cv_r2 = float(cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean())

        mlflow.log_metrics({
            "mse":   mse,
            "rmse":  rmse,
            "mae":   mae,
            "r2":    r2,
            "cv_r2": cv_r2,
        })

        mlflow.sklearn.log_model(
            pipe,
            name="model",
            registered_model_name="MLProjectModel",
            input_example=X_test[:3],
        )

        print(f"MSE={mse:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  CV-R²={cv_r2:.4f}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ridge regression via MLflow Projects")
    parser.add_argument("--n_samples",   type=int,   default=300)
    parser.add_argument("--noise_level", type=float, default=0.5)
    parser.add_argument("--test_size",   type=float, default=0.2)
    parser.add_argument("--alpha",       type=float, default=1.0)
    args = parser.parse_args()
    main(args.n_samples, args.noise_level, args.test_size, args.alpha)
