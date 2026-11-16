"""
================================================================================
MODULE 10: Hyperparameter Tuning with MLflow — Nested Runs & Best Model
================================================================================
MLflow's nested run model is a natural fit for hyperparameter tuning:
  - Parent run = the whole tuning experiment (stores the winner)
  - Child runs  = individual trials (each gets its own params + metrics)

This gives you:
  - A clean hierarchy in the UI (one parent per sweep)
  - Full auditability (every trial is inspectable)
  - Programmatic best-run selection via search_runs()
  - Automatic promotion of the winner to the Registry

This module covers:
  A. Manual grid search with nested runs
  B. Random search with nested runs
  C. Finding and promoting the best child run
  D. A reusable tuning helper pattern
================================================================================
"""

import itertools

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Data ──────────────────────────────────────────────────────────────────────
np.random.seed(42)
n = 600
X = np.random.randn(n, 5)
true_w = np.array([3.0, -1.5, 2.0, 0.5, 1.0])
y = X @ true_w + 2 + np.random.randn(n) * 0.4

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("10_hyperparameter_tuning")
client = MlflowClient()
REGISTERED_NAME = "TunedModel"


# ══════════════════════════════════════════════════════════════════════════════
# A. Grid search — Ridge alpha sweep (nested runs)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── A. Ridge alpha grid search ──")

alpha_grid = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

with mlflow.start_run(run_name="sweep_ridge_alpha") as parent_run:
    mlflow.set_tags({
        "sweep_type":  "grid",
        "model_class": "Ridge",
        "param":       "alpha",
    })
    mlflow.log_param("alpha_grid", str(alpha_grid))

    best_mse   = float("inf")
    best_alpha = None

    for alpha in alpha_grid:
        with mlflow.start_run(run_name=f"ridge_alpha_{alpha}", nested=True):
            pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=alpha))])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            mse  = mean_squared_error(y_test, y_pred)
            r2   = r2_score(y_test, y_pred)
            cv   = float(cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean())

            mlflow.log_params({"alpha": alpha, "model_type": "Ridge"})
            mlflow.log_metrics({"mse": mse, "r2": r2, "cv_r2": cv})
            mlflow.sklearn.log_model(pipe, "model")

            marker = " ← best" if mse < best_mse else ""
            print(f"    alpha={alpha:>7}  MSE={mse:.4f}  R²={r2:.4f}  CV-R²={cv:.4f}{marker}")

            if mse < best_mse:
                best_mse   = mse
                best_alpha = alpha

    # Log the winner to the parent
    mlflow.log_metrics({"best_mse": best_mse})
    mlflow.log_params({"best_alpha": best_alpha})
    parent_ridge_id = parent_run.info.run_id
    print(f"  → Best alpha={best_alpha}  MSE={best_mse:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# B. Grid search — RandomForest 2D grid (n_estimators × max_depth)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── B. RandomForest 2D grid search ──")

n_estimators_grid = [50, 100, 200]
max_depth_grid    = [4, 6, 8, None]

rf_grid = list(itertools.product(n_estimators_grid, max_depth_grid))
print(f"  Grid size: {len(rf_grid)} trials")

with mlflow.start_run(run_name="sweep_rf_2d_grid") as parent_rf:
    mlflow.set_tags({"sweep_type": "grid", "model_class": "RandomForest"})
    mlflow.log_param("grid_size", len(rf_grid))

    best_rf_mse    = float("inf")
    best_rf_params = {}

    for n_est, max_d in rf_grid:
        trial_name = f"rf_n{n_est}_d{max_d if max_d else 'None'}"
        with mlflow.start_run(run_name=trial_name, nested=True):
            rf = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2  = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            mlflow.log_params({"n_estimators": n_est, "max_depth": str(max_d), "model_type": "RandomForest"})
            mlflow.log_metrics({"mse": mse, "r2": r2, "mae": mae})
            mlflow.sklearn.log_model(rf, "model")

            if mse < best_rf_mse:
                best_rf_mse    = mse
                best_rf_params = {"n_estimators": n_est, "max_depth": max_d}

    mlflow.log_metrics({"best_mse": best_rf_mse})
    mlflow.log_params(best_rf_params)
    parent_rf_id = parent_rf.info.run_id
    print(f"  → Best RF: {best_rf_params}  MSE={best_rf_mse:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# C. Random search — GradientBoosting with random sampling
# ══════════════════════════════════════════════════════════════════════════════
print("\n── C. GradientBoosting random search (20 trials) ──")

rng = np.random.default_rng(42)
N_TRIALS = 20

with mlflow.start_run(run_name="sweep_gb_random") as parent_gb:
    mlflow.set_tags({"sweep_type": "random", "model_class": "GradientBoosting"})
    mlflow.log_param("n_trials", N_TRIALS)

    best_gb_mse    = float("inf")
    best_gb_params = {}

    for trial in range(N_TRIALS):
        params = {
            "n_estimators":    int(rng.integers(50, 300)),
            "max_depth":       int(rng.integers(2, 8)),
            "learning_rate":   float(rng.uniform(0.01, 0.3)),
            "subsample":       float(rng.uniform(0.6, 1.0)),
            "min_samples_leaf": int(rng.integers(1, 20)),
        }

        with mlflow.start_run(run_name=f"gb_trial_{trial:02d}", nested=True):
            gb = GradientBoostingRegressor(**params, random_state=42)
            gb.fit(X_train, y_train)
            y_pred = gb.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2  = r2_score(y_test, y_pred)

            mlflow.log_params({**params, "model_type": "GradientBoosting"})
            mlflow.log_metrics({"mse": mse, "r2": r2})
            mlflow.sklearn.log_model(gb, "model")

            if mse < best_gb_mse:
                best_gb_mse    = mse
                best_gb_params = params

    mlflow.log_metrics({"best_mse": best_gb_mse})
    mlflow.log_params(best_gb_params)
    parent_gb_id = parent_gb.info.run_id
    print(f"  → Best GB: n_est={best_gb_params['n_estimators']}  "
          f"depth={best_gb_params['max_depth']}  lr={best_gb_params['learning_rate']:.3f}  "
          f"MSE={best_gb_mse:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# D. Find global best child run across all sweeps, register it
# ══════════════════════════════════════════════════════════════════════════════
print("\n── D. Selecting and registering the overall best model ──")

exp = mlflow.get_experiment_by_name("10_hyperparameter_tuning")

# All finished runs in the experiment
all_runs = mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string='status = "FINISHED" AND metrics.mse > 0',
    order_by=["metrics.mse ASC"],
)

# Exclude parent runs (they log best_mse, not mse from training)
# Parent runs have "sweep_type" tag; child runs don't
child_runs = all_runs[
    all_runs["tags.sweep_type"].isna() | (all_runs["tags.sweep_type"] == "")
].dropna(subset=["metrics.mse"])

if len(child_runs) == 0:
    # Fallback: just use all runs
    child_runs = all_runs

best_row    = child_runs.loc[child_runs["metrics.mse"].idxmin()]
best_run_id = best_row["run_id"]
best_global = best_row["metrics.mse"]
best_r2     = best_row.get("metrics.r2", float("nan"))

print(f"  Global best:  MSE={best_global:.4f}  R²={best_r2:.4f}")
print(f"  Run ID:       {best_run_id[:20]}...")

# Register the best model
registered = mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name=REGISTERED_NAME,
)
client.set_registered_model_alias(REGISTERED_NAME, "champion", registered.version)
client.update_model_version(
    name=REGISTERED_NAME,
    version=registered.version,
    description=f"Best tuned model  MSE={best_global:.4f}  promoted by 10-hyperparameter sweep",
)
print(f"  Registered:   {REGISTERED_NAME}  v{registered.version}  @champion")


# ══════════════════════════════════════════════════════════════════════════════
# E. Summary table — top 5 runs
# ══════════════════════════════════════════════════════════════════════════════
print("\n── E. Top 5 child runs by MSE ──")

top5_cols = ["tags.mlflow.runName", "metrics.mse", "metrics.r2"]
available = [c for c in top5_cols if c in child_runs.columns]
top5 = child_runs.sort_values("metrics.mse").head(5)[available]
top5.columns = [c.split(".")[-1] for c in available]
print(top5.to_string(index=False))


print()
print("=" * 64)
print("NEXT: mlflow ui  →  http://localhost:5000")
print("      Open '10_hyperparameter_tuning'.")
print("      Each sweep parent run has nested child runs — expand them.")
print("      Compare across sweeps: Ridge vs RF vs GradientBoosting.")
print("=" * 64)
print()
print("KEY TAKEAWAYS:")
print("  nested=True           → child runs under the active parent run")
print("  Parent stores summary; each child stores one trial's full details")
print("  search_runs() + order_by='metrics.mse ASC' finds the winner")
print("  register_model() + set_registered_model_alias() promotes the best")
print("  Random search often finds good configs faster than grid search")
print("  The UI lets you compare all child runs across sweeps side-by-side")
