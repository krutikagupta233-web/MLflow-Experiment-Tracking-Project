"""
================================================================================
MODULE 3: MLflow Autologging — Zero-Effort Experiment Tracking
================================================================================
One call before model.fit() automatically logs: params, metrics, the model
itself, input examples, and model signatures — with zero manual log_* calls.

This module covers:
  A. Autolog for sklearn — what gets captured
  B. What autolog logs vs what you add manually
  C. Autolog with cross-validation (GridSearchCV)
  D. Fine-grained control: enabling, disabling, per-framework options
================================================================================
"""

import mlflow
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

# ── Data ──────────────────────────────────────────────────────────────────────
np.random.seed(42)
X = np.random.randn(300, 5)
true_w = np.array([2.5, 1.5, 3.0, 0.5, 1.0])
y = X @ true_w + 1 + np.random.randn(300) * 0.5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("03_autologging")


# ══════════════════════════════════════════════════════════════════════════════
# A. Autolog with basic models — compare captured columns vs manual logging
# ══════════════════════════════════════════════════════════════════════════════
print("\n── A. Autolog vs manual logging ──")

# ------ Run 1: Manual logging (baseline for comparison) ------
with mlflow.start_run(run_name="manual_logging") as run:
    m = Ridge(alpha=1.0)
    m.fit(X_train, y_train)

    mlflow.log_params({"model": "Ridge", "alpha": 1.0})
    mlflow.log_metrics({
        "mse": mean_squared_error(y_test, m.predict(X_test)),
        "r2":  r2_score(y_test, m.predict(X_test)),
    })
    mlflow.sklearn.log_model(m, "model")
    manual_run_id = run.info.run_id
    print(f"  Manual run: {manual_run_id[:12]}...")

# ------ Run 2: Autolog (same model, zero manual calls) ------
mlflow.sklearn.autolog(
    log_models=True,           # save the trained sklearn model
    log_datasets=True,         # log a profile of the training dataset
    log_input_examples=True,   # save sample rows so the model is self-documenting
    log_model_signatures=True, # save input/output schema (critical for serving)
)

with mlflow.start_run(run_name="autolog_ridge") as run:
    m2 = Ridge(alpha=1.0)
    m2.fit(X_train, y_train)   # autolog fires here

    # Add custom metrics ON TOP of what autolog already captured
    mlflow.log_metric("custom_test_mse", mean_squared_error(y_test, m2.predict(X_test)))
    mlflow.log_metric("custom_test_r2",  r2_score(y_test, m2.predict(X_test)))
    auto_run_id = run.info.run_id
    print(f"  Autolog run: {auto_run_id[:12]}...")

mlflow.sklearn.autolog(disable=True)

# Show what autolog captured that manual didn't
from mlflow.tracking import MlflowClient
client = MlflowClient()
manual_cols = set(client.get_run(manual_run_id).data.params.keys())
auto_cols   = set(client.get_run(auto_run_id).data.params.keys())
extra = auto_cols - manual_cols
if extra:
    print(f"\n  Extra params autolog captured (vs manual): {sorted(extra)}")


# ══════════════════════════════════════════════════════════════════════════════
# B. Autolog with RandomForest — feature importances are captured automatically
# ══════════════════════════════════════════════════════════════════════════════
print("\n── B. RandomForest with autolog ──")

mlflow.sklearn.autolog(log_models=True, log_input_examples=True)

feature_names = ["f0", "f1", "f2", "f3", "f4"]

for n_estimators in [50, 100, 200]:
    with mlflow.start_run(run_name=f"rf_{n_estimators}_trees"):
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)

        mse = mean_squared_error(y_test, rf.predict(X_test))
        r2  = r2_score(y_test, rf.predict(X_test))

        # Autolog captures n_estimators, max_depth, etc. automatically
        # We add the explicit test metrics
        mlflow.log_metrics({"test_mse": mse, "test_r2": r2})

        # Manually log feature importances as individual metrics
        for name, importance in zip(feature_names, rf.feature_importances_):
            mlflow.log_metric(f"importance_{name}", importance)

        print(f"  RF n={n_estimators:3d}  MSE={mse:.4f}  R²={r2:.4f}")

mlflow.sklearn.autolog(disable=True)


# ══════════════════════════════════════════════════════════════════════════════
# C. Autolog with GridSearchCV — logs every CV trial as a child run
# ══════════════════════════════════════════════════════════════════════════════
print("\n── C. GridSearchCV with autolog ──")

# max_tuning_runs=None logs ALL CV trials; set a number to cap child runs
mlflow.sklearn.autolog(
    log_models=True,
    max_tuning_runs=5,         # log only the top-5 trials to avoid clutter
    log_post_training_metrics=True,
)

with mlflow.start_run(run_name="gridsearch_ridge"):
    param_grid = {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    gs = GridSearchCV(Ridge(), param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    gs.fit(X_train, y_train)

    best_alpha = gs.best_params_["alpha"]
    best_mse   = -gs.best_score_
    print(f"  Best alpha={best_alpha}  CV-MSE={best_mse:.4f}")
    print(f"  Best estimator: {gs.best_estimator_}")

mlflow.sklearn.autolog(disable=True)


# ══════════════════════════════════════════════════════════════════════════════
# D. Show everything that was logged
# ══════════════════════════════════════════════════════════════════════════════
print("\n── D. Full column inventory from autolog ──")

exp = mlflow.get_experiment_by_name("03_autologging")
runs_df = mlflow.search_runs([exp.experiment_id])

metric_cols = [c for c in runs_df.columns if c.startswith("metrics.")]
param_cols  = [c for c in runs_df.columns if c.startswith("params.")]
print(f"  Metric columns ({len(metric_cols)}): {sorted(metric_cols)}")
print(f"  Param columns  ({len(param_cols)}): {sorted(param_cols)}")


print()
print("=" * 62)
print("NEXT: mlflow ui  →  http://localhost:5000")
print("      Open '03_autologging'. Notice:")
print("      - autolog_ridge has a model signature and input example")
print("      - RF runs have feature importance metrics logged")
print("      - GridSearchCV run has child runs for each CV trial")
print("=" * 62)
print()
print("KEY TAKEAWAYS:")
print("  autolog() fires on model.fit() — call it BEFORE training")
print("  log_models=True        → saves model + signature + input example")
print("  log_input_examples=True → makes the model self-documenting for serving")
print("  max_tuning_runs=N      → caps child runs from GridSearchCV")
print("  autolog(disable=True)  → turn off when done")
print("  You can always add custom log_metric() calls alongside autolog")
print("  Supported: sklearn, xgboost, lightgbm, pytorch, tensorflow, statsmodels")
