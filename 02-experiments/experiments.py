"""
================================================================================
MODULE 2: MLflow Experiments — Organise, Search, and Compare Runs
================================================================================
An experiment groups related runs so you can compare them side-by-side in the
UI and query them programmatically.

This module covers:
  A. Creating experiments and running multiple models under one experiment
  B. Programmatic search — filter, sort, and inspect runs as a DataFrame
  C. Using MlflowClient for low-level run inspection
================================================================================
"""

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Data ──────────────────────────────────────────────────────────────────────
np.random.seed(42)
n = 400
X = np.random.randn(n, 4)
true_w = np.array([3.0, -1.5, 2.0, 0.5])
y = X @ true_w + 2 + np.random.randn(n) * 0.4

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CREATE AN EXPERIMENT (idempotent — reuses if already exists)
# ══════════════════════════════════════════════════════════════════════════════
EXPERIMENT_NAME = "02_model_comparison"

try:
    exp_id = mlflow.create_experiment(
        name=EXPERIMENT_NAME,
        tags={
            "purpose": "compare_regression_models",
            "dataset": "synthetic_4_features",
            "author":  "mlflow_learner",
        },
    )
    print(f"Created experiment '{EXPERIMENT_NAME}'  id={exp_id}")
except mlflow.exceptions.MlflowException:
    exp_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    print(f"Reusing experiment '{EXPERIMENT_NAME}'  id={exp_id}")

mlflow.set_experiment(EXPERIMENT_NAME)


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAIN MULTIPLE MODELS — each run logged to the same experiment
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Training models ──")

model_configs = {
    "LinearRegression":   LinearRegression(),
    "Ridge_alpha_1.0":    Ridge(alpha=1.0),
    "Ridge_alpha_0.1":    Ridge(alpha=0.1),
    "Lasso_alpha_0.1":    Lasso(alpha=0.1, max_iter=10000),
    "RandomForest_100":   RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
    "GradientBoost_100":  GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
}

for name, base_model in model_configs.items():
    with mlflow.start_run(run_name=name) as run:
        # Wrap in a pipeline so scaling is part of the model
        pipe = Pipeline([("scaler", StandardScaler()), ("model", base_model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mse  = mean_squared_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = mse ** 0.5

        # 5-fold CV score for a more honest estimate
        cv_r2 = float(cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean())

        mlflow.log_params({
            "model_class":   type(base_model).__name__,
            "n_features":    X.shape[1],
            "train_samples": len(X_train),
            "pipeline":      "StandardScaler + model",
        })
        mlflow.log_metrics({
            "mse":    mse,
            "rmse":   rmse,
            "mae":    mae,
            "r2":     r2,
            "cv_r2":  cv_r2,
        })
        mlflow.set_tag("model_family", name.split("_")[0])
        mlflow.sklearn.log_model(pipe, "model")

        print(f"  {name:25s}  MSE={mse:.4f}  R²={r2:.4f}  CV-R²={cv_r2:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. SEARCH RUNS PROGRAMMATICALLY
#    search_runs() returns a pandas DataFrame — standard pandas operations apply
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Programmatic search ──")

# All runs in this experiment
all_runs = mlflow.search_runs(experiment_ids=[exp_id])
print(f"Total runs: {len(all_runs)}")

# Filter: only runs where R² > 0.98
high_r2 = mlflow.search_runs(
    experiment_ids=[exp_id],
    filter_string='metrics.r2 > 0.98',
    order_by=["metrics.r2 DESC"],
)
print(f"Runs with R² > 0.98: {len(high_r2)}")

# Filter: only runs for a specific model family
tree_runs = mlflow.search_runs(
    experiment_ids=[exp_id],
    filter_string='tags.model_family = "RandomForest" OR tags.model_family = "GradientBoost"',
    order_by=["metrics.mse ASC"],
)
print(f"Tree-based model runs: {len(tree_runs)}")

# Best run by MSE
best_row = all_runs.loc[all_runs["metrics.mse"].idxmin()]
print(f"\nBest by MSE:  {best_row['tags.mlflow.runName']}  (MSE={best_row['metrics.mse']:.4f})")

best_row_r2 = all_runs.loc[all_runs["metrics.r2"].idxmax()]
print(f"Best by R²:   {best_row_r2['tags.mlflow.runName']}  (R²={best_row_r2['metrics.r2']:.4f})")

# Top 3 models ranked by MSE
print("\nTop 3 by MSE:")
top3 = all_runs.sort_values("metrics.mse").head(3)[
    ["tags.mlflow.runName", "metrics.mse", "metrics.r2", "metrics.cv_r2"]
]
top3.columns = ["Model", "MSE", "R²", "CV-R²"]
print(top3.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 4. MLFLOWCLIENT — low-level access to run details
# ══════════════════════════════════════════════════════════════════════════════
print("\n── MlflowClient inspection ──")

client = MlflowClient()
best_run_id = best_row["run_id"]
run_obj = client.get_run(best_run_id)

print(f"Run name:   {run_obj.data.tags.get('mlflow.runName')}")
print(f"Status:     {run_obj.info.status}")
print(f"Params:     {run_obj.data.params}")
print(f"Metrics:    { {k: f'{v:.4f}' for k, v in run_obj.data.metrics.items()} }")

# List all experiments
print("\nAll experiments:")
for exp in client.search_experiments():
    print(f"  id={exp.experiment_id}  name={exp.name}")


print()
print("=" * 62)
print("NEXT: mlflow ui  →  http://localhost:5000")
print("      Click '02_model_comparison' → select all runs → Compare.")
print("      Sort columns by MSE or R² to rank models instantly.")
print("=" * 62)
print()
print("KEY TAKEAWAYS:")
print("  create_experiment()    — group related runs (idempotent wrapper needed)")
print("  set_experiment()       — direct all subsequent runs here")
print("  search_runs()          — returns a DataFrame (pandas operations work)")
print("  filter_string          — SQL-like: 'metrics.mse < 0.1 AND tags.env = \"prod\"'")
print("  order_by               — sort results: ['metrics.mse ASC']")
print("  MlflowClient           — low-level access to any run, experiment, or artifact")
