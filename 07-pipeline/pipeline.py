"""
================================================================================
MODULE 7: End-to-End MLflow Pipeline
================================================================================
A realistic production workflow:
  1. Generate data
  2. Train multiple models
  3. Track every run with MLflow
  4. Register all models in the Registry
  5. Promote the best model with an alias
  6. Load the champion and make predictions

This module fixes two issues from naive implementations:
  - Version numbers are looked up by run_id (never hardcoded)
  - Aliases are used instead of deprecated stage transitions
================================================================================
"""

import time
import warnings

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

print("=" * 64)
print("  MLFLOW END-TO-END PIPELINE")
print("=" * 64)


# ══════════════════════════════════════════════════════════════════════════════
# 1. SETUP
# ══════════════════════════════════════════════════════════════════════════════
EXPERIMENT_NAME = "07_end_to_end_pipeline"
REGISTERED_NAME = "PipelineModel"

try:
    mlflow.create_experiment(EXPERIMENT_NAME, tags={"pipeline": "end_to_end", "module": "7"})
except mlflow.exceptions.MlflowException:
    pass
mlflow.set_experiment(EXPERIMENT_NAME)

client = MlflowClient()
print(f"\n[1/6] Experiment: {EXPERIMENT_NAME}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA
# ══════════════════════════════════════════════════════════════════════════════
np.random.seed(42)
n = 600
X = np.random.randn(n, 5)
true_w = np.array([3.0, -1.5, 2.0, 0.0, 0.5])   # feature 3 is pure noise
y = X @ true_w + 2 + np.random.randn(n) * 0.3

feature_names = ["feat_a", "feat_b", "feat_c", "feat_noise", "feat_e"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
eval_df = pd.DataFrame(X_test, columns=feature_names)
eval_df["target"] = y_test

print(f"[2/6] Data: {n} samples  {X.shape[1]} features  "
      f"train={len(X_train)}  test={len(X_test)}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN MULTIPLE MODELS — each logged to the same experiment
# ══════════════════════════════════════════════════════════════════════════════
print(f"[3/6] Training models...")

model_configs = {
    "LinearRegression":   LinearRegression(),
    "Ridge_alpha_1":      Ridge(alpha=1.0),
    "Ridge_alpha_0.1":    Ridge(alpha=0.1),
    "Lasso_alpha_0.1":    Lasso(alpha=0.1, max_iter=10000),
    "RandomForest_100":   RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
    "GradientBoost_100":  GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
}

best_run_id = None
best_mse    = float("inf")

for name, base_model in model_configs.items():
    with mlflow.start_run(run_name=name, tags={"model_family": name.split("_")[0]}) as run:
        pipe = Pipeline([("scaler", StandardScaler()), ("model", base_model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mse  = mean_squared_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = mse ** 0.5

        mlflow.log_params({
            "model":          name,
            "model_class":    type(base_model).__name__,
            "n_features":     X.shape[1],
            "train_samples":  len(X_train),
        })
        mlflow.log_metrics({"mse": mse, "rmse": rmse, "r2": r2, "mae": mae})

        # Feature importances for tree-based models
        if hasattr(base_model, "feature_importances_"):
            for feat, imp in zip(feature_names, base_model.feature_importances_):
                mlflow.log_metric(f"imp_{feat}", imp)

        # Register every model — versions accumulate in the Registry
        mlflow.sklearn.log_model(
            pipe,
            "model",
            registered_model_name=REGISTERED_NAME,
            input_example=pd.DataFrame(X_test[:3], columns=feature_names),
        )

        marker = "  ← BEST" if mse < best_mse else ""
        print(f"    {name:25s}  MSE={mse:.4f}  R²={r2:.4f}{marker}")

        if mse < best_mse:
            best_mse   = mse
            best_run_id = run.info.run_id


# ══════════════════════════════════════════════════════════════════════════════
# 4. REGISTRY — promote best model using aliases (no hardcoded versions)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[4/6] Promoting best model...")

# Wait for registration writes to complete
time.sleep(1)

# Look up version by run_id — never hardcode version numbers
version_map = {mv.run_id: mv.version
               for mv in client.search_model_versions(f"name='{REGISTERED_NAME}'")}

champion_version = version_map.get(best_run_id)
if champion_version:
    # Use alias (modern approach)
    client.set_registered_model_alias(REGISTERED_NAME, "champion", champion_version)
    client.update_model_version(
        name=REGISTERED_NAME,
        version=champion_version,
        description=f"Best model — MSE={best_mse:.4f}. Auto-promoted by pipeline.",
    )
    client.set_model_version_tag(REGISTERED_NAME, champion_version, "approved_by", "pipeline")
    print(f"    v{champion_version} → @champion  (MSE={best_mse:.4f})")
else:
    print("    WARNING: Could not find champion version — registration may still be pending")


# ══════════════════════════════════════════════════════════════════════════════
# 5. LOAD AND PREDICT
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[5/6] Loading @champion and making predictions...")

try:
    champion = mlflow.sklearn.load_model(f"models:/{REGISTERED_NAME}@champion")
except Exception:
    # Fallback: load by run_id if alias isn't ready yet
    champion = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")

new_inputs = np.array([
    [ 1.0, -0.5,  2.0,  1.5,  0.0],
    [ 0.0,  0.0,  0.0,  0.0,  0.0],
    [-1.0,  2.0, -1.0,  0.5,  0.5],
    [ 0.5,  1.0, -0.5, -1.0,  2.0],
])
predictions = champion.predict(new_inputs)
expected    = new_inputs @ true_w + 2

for i, (pred, exp) in enumerate(zip(predictions, expected)):
    delta = abs(pred - exp)
    print(f"    Row {i+1}: predicted={pred:8.4f}  expected={exp:8.4f}  Δ={delta:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[6/6] Pipeline complete!")
print()
print(f"{'=' * 64}")
print(f"  Experiment: {EXPERIMENT_NAME}")
print(f"  Models trained: {len(model_configs)}")
print(f"  Best MSE:   {best_mse:.4f}")
print(f"  Registered: {REGISTERED_NAME}  v{champion_version} (@champion)")
print(f"{'=' * 64}")
print()
print("COMMANDS:")
print(f"  mlflow ui")
print(f"  mlflow models serve -m 'models:/{REGISTERED_NAME}@champion' --port 5001 --no-conda")
print()
print("KEY TAKEAWAYS:")
print("  A real pipeline: train → track → register → promote → predict")
print("  Use version_map = {mv.run_id: mv.version} — never hardcode versions")
print("  set_registered_model_alias() is the modern alternative to stage transitions")
print("  Feature importances can be logged as per-feature metrics")
print("  The entire history is preserved — every run is auditable")
