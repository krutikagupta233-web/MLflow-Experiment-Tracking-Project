"""
================================================================================
MODULE 8: MLflow Evaluate — Systematic Model Assessment
================================================================================
mlflow.evaluate() goes beyond logging metrics manually. It:
  - Runs your model against an evaluation dataset automatically
  - Computes a standard set of built-in metrics (RMSE, R², MAE, MAPE, ...)
  - Saves evaluation artifacts (residual plots, metric tables) to the run
  - Lets you define custom metrics alongside the built-in ones
  - Enables baseline comparison (is my model better than a naive baseline?)

This module covers:
  A. Basic mlflow.evaluate() for regression
  B. Custom metrics
  C. Baseline comparison
  D. Comparing multiple models via evaluate()
================================================================================
"""

import mlflow
import numpy as np
import pandas as pd
from mlflow.metrics import MetricValue, make_metric
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ── Data ──────────────────────────────────────────────────────────────────────
np.random.seed(42)
n = 500
X = np.random.randn(n, 4)
true_w = np.array([3.0, -1.5, 2.0, 0.5])
y = X @ true_w + 2 + np.random.randn(n) * 0.4

feature_cols = ["feat_0", "feat_1", "feat_2", "feat_3"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# mlflow.evaluate() expects a DataFrame with BOTH features AND target column
eval_df = pd.DataFrame(X_test, columns=feature_cols)
eval_df["target"] = y_test

mlflow.set_experiment("08_evaluation")
client = MlflowClient()


# ══════════════════════════════════════════════════════════════════════════════
# A. Train models and log them (evaluation is separate from training)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── A. Training models ──")

REGISTERED_NAME = "EvalModel"
model_run_ids = {}

models = {
    "LinearRegression": LinearRegression(),
    "Ridge_1.0":        Ridge(alpha=1.0),
    "RandomForest_100": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "GradientBoost":    GradientBoostingRegressor(n_estimators=100, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    with mlflow.start_run(run_name=f"train_{name}") as run:
        mlflow.log_param("model", name)
        mlflow.log_metric("train_r2", model.score(X_train, y_train))
        mlflow.sklearn.log_model(
            model, "model",
            input_example=pd.DataFrame(X_test[:3], columns=feature_cols),
        )
        model_run_ids[name] = run.info.run_id
    print(f"  {name:25s}  trained  run={run.info.run_id[:12]}...")


# ══════════════════════════════════════════════════════════════════════════════
# B. Define a custom metric
#    Built-in regression metrics: mae, mse, rmse, r2, mape, max_error
#    Add a custom one: fraction of predictions within 10% of actual value
# ══════════════════════════════════════════════════════════════════════════════
def within_10pct_fn(predictions, targets, metrics, sample_weights=None):
    """Fraction of predictions where |pred - actual| / |actual| <= 10%."""
    preds = np.array(predictions)
    acts  = np.array(targets)
    denom = np.where(np.abs(acts) > 0.01, np.abs(acts), 0.01)
    within = (np.abs(acts - preds) / denom) <= 0.10
    return MetricValue(
        scores=list(within.astype(float)),
        aggregate_results={"mean": float(within.mean())},
    )

within_10pct = make_metric(
    eval_fn=within_10pct_fn,
    name="within_10pct",
    greater_is_better=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# C. Evaluate each model with mlflow.evaluate()
#    Each evaluation run logs: built-in metrics + custom metric + artifacts
# ══════════════════════════════════════════════════════════════════════════════
print("\n── B. Evaluating with mlflow.evaluate() ──")

eval_results = {}

for name, train_run_id in model_run_ids.items():
    model_uri = f"runs:/{train_run_id}/model"

    with mlflow.start_run(run_name=f"eval_{name}") as eval_run:
        result = mlflow.evaluate(
            model=model_uri,
            data=eval_df,
            targets="target",
            model_type="regressor",          # auto-computes: mae, mse, rmse, r2, mape
            extra_metrics=[within_10pct],    # our custom metric
        )
        eval_results[name] = result.metrics

        rmse       = result.metrics.get("root_mean_squared_error", float("nan"))
        r2         = result.metrics.get("r2_score", float("nan"))
        mape       = result.metrics.get("mean_absolute_percentage_error", float("nan"))
        w10        = result.metrics.get("within_10pct/mean", float("nan"))
        print(f"  {name:25s}  RMSE={rmse:.4f}  R²={r2:.4f}  MAPE={mape:.1f}%  within10%={w10:.2%}")


# ══════════════════════════════════════════════════════════════════════════════
# D. Baseline comparison
#    mlflow.evaluate() can compare your model against a simple baseline
#    Baseline = a Python callable (e.g., always predict the mean)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── C. Baseline comparison ──")

mean_prediction = y_train.mean()

def baseline_mean(model_input):
    """Naive baseline: always predict the training mean."""
    return np.full(len(model_input), mean_prediction)

with mlflow.start_run(run_name="eval_baseline_mean"):
    baseline_result = mlflow.evaluate(
        model=baseline_mean,                # pass a callable directly
        data=eval_df,
        targets="target",
        model_type="regressor",
        extra_metrics=[within_10pct],
    )
    b_rmse = baseline_result.metrics.get("root_mean_squared_error", float("nan"))
    b_r2   = baseline_result.metrics.get("r2_score", float("nan"))
    print(f"  Baseline (mean):           RMSE={b_rmse:.4f}  R²={b_r2:.4f}")

# Now compare best model to baseline
best_name = min(eval_results, key=lambda k: eval_results[k].get("root_mean_squared_error", float("inf")))
best_rmse  = eval_results[best_name].get("root_mean_squared_error", float("nan"))
improvement = (b_rmse - best_rmse) / b_rmse * 100
print(f"\n  Best model ({best_name}) vs baseline:")
print(f"    RMSE improvement: {improvement:.1f}%  ({b_rmse:.4f} → {best_rmse:.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# E. Summary: what mlflow.evaluate() logged automatically
# ══════════════════════════════════════════════════════════════════════════════
print("\n── D. What evaluate() logged ──")
print("  Each evaluation run contains:")
print("  Metrics: root_mean_squared_error, mean_absolute_error, r2_score,")
print("           mean_absolute_percentage_error, max_error, within_10pct/mean")
print("  Artifacts: eval_results_table.json (row-level predictions vs actuals)")
print()
print("  View in UI: mlflow ui → '08_evaluation' → any eval_* run → Metrics + Artifacts")


print()
print("=" * 64)
print("NEXT: mlflow ui  →  http://localhost:5000")
print("      Open '08_evaluation'. Compare eval_* runs side by side.")
print("      Click any eval run → Artifacts → eval_results_table.json")
print("=" * 64)
print()
print("KEY TAKEAWAYS:")
print("  mlflow.evaluate(model, data, targets, model_type)")
print("    → automatically computes all standard regression/classifier metrics")
print("    → logs metric table artifact with per-row predictions")
print("  extra_metrics=[custom_metric]  → add your own MetricValue-returning fn")
print("  Pass a callable as model= to evaluate a baseline (no logging needed)")
print("  Separation of training and evaluation runs is a best practice")
