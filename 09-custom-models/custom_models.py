"""
================================================================================
MODULE 9: MLflow Custom Models — pyfunc Wrappers
================================================================================
`mlflow.pyfunc` is MLflow's generic model interface. It lets you wrap ANY
Python object as a loadable MLflow model — including models that don't have
a native MLflow flavor (custom torch, statsmodels, rule-based models, etc.)

The key abstraction: subclass mlflow.pyfunc.PythonModel and implement:
  - load_context(context)  — load artifacts when the model is loaded
  - predict(context, model_input)  — run inference

This module covers:
  A. A preprocessed model (scaler + sklearn model, bundled together)
  B. A text preprocessor (custom tokeniser + model — shows non-sklearn use case)
  C. Loading and serving the pyfunc model
================================================================================
"""

import os
import pickle
import tempfile

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Data ──────────────────────────────────────────────────────────────────────
np.random.seed(42)
X = np.random.randn(400, 4)
y = 3.0 * X[:, 0] - 1.5 * X[:, 1] + 2.0 * X[:, 2] + 0.5 * X[:, 3] + 2 + np.random.randn(400) * 0.3
feature_cols = ["feat_0", "feat_1", "feat_2", "feat_3"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("09_custom_models")


# ══════════════════════════════════════════════════════════════════════════════
# A. Custom pyfunc: scaler + model bundled together
#    Normally mlflow.sklearn only saves the model. If you want preprocessing
#    baked in (for serving safety), wrap both in a pyfunc.
# ══════════════════════════════════════════════════════════════════════════════

class ScaledModel(mlflow.pyfunc.PythonModel):
    """
    Bundles a StandardScaler and a Ridge regressor.
    predict() applies scaling then inference — no separate preprocessing step.
    """

    def load_context(self, context):
        with open(context.artifacts["scaler"], "rb") as f:
            self.scaler = pickle.load(f)
        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            X = model_input.values
        else:
            X = np.array(model_input)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


print("\n── A. Preprocessed pyfunc model ──")

# Train the components separately
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

ridge = Ridge(alpha=0.5)
ridge.fit(X_train_scaled, y_train)

preds_check = ridge.predict(scaler.transform(X_test))
mse = mean_squared_error(y_test, preds_check)
r2  = r2_score(y_test, preds_check)
print(f"  Ridge + Scaler  MSE={mse:.4f}  R²={r2:.4f}")

# Serialize components to temp files, log as artifacts
with tempfile.TemporaryDirectory() as tmpdir:
    scaler_path = os.path.join(tmpdir, "scaler.pkl")
    model_path  = os.path.join(tmpdir, "model.pkl")

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(model_path, "wb") as f:
        pickle.dump(ridge, f)

    with mlflow.start_run(run_name="pyfunc_scaled_ridge") as run:
        mlflow.log_params({"model_type": "Ridge", "alpha": 0.5, "scaler": "StandardScaler"})
        mlflow.log_metrics({"mse": mse, "r2": r2})

        # Log the pyfunc model — artifacts dict maps name → local path
        mlflow.pyfunc.log_model(
            artifact_path="scaled_model",
            python_model=ScaledModel(),
            artifacts={
                "scaler": scaler_path,
                "model":  model_path,
            },
            input_example=pd.DataFrame(X_test[:3], columns=feature_cols),
        )
        scaled_model_uri = f"runs:/{run.info.run_id}/scaled_model"
        print(f"  Logged pyfunc model → {scaled_model_uri[:50]}...")


# ══════════════════════════════════════════════════════════════════════════════
# B. Custom pyfunc with domain logic: feature engineering inside predict()
#    Shows how to put custom preprocessing logic inside the model itself
# ══════════════════════════════════════════════════════════════════════════════

class FeatureEngineeringModel(mlflow.pyfunc.PythonModel):
    """
    Adds polynomial interaction features before predicting.
    Bundles feature engineering with the model — the caller just passes raw features.
    """

    def load_context(self, context):
        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)
        with open(context.artifacts["feature_names"], "rb") as f:
            self.feature_names = pickle.load(f)

    def _add_interactions(self, X: np.ndarray) -> np.ndarray:
        """Add pairwise product features to the raw input."""
        n_orig = X.shape[1]
        interactions = []
        for i in range(n_orig):
            for j in range(i + 1, n_orig):
                interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))
        return np.hstack([X] + interactions)

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            X = model_input.values
        else:
            X = np.array(model_input)
        X_engineered = self._add_interactions(X)
        return self.model.predict(X_engineered)


print("\n── B. Feature engineering pyfunc model ──")

# Build interaction features for training
n_orig = X_train.shape[1]
interactions_train = []
interactions_test  = []
for i in range(n_orig):
    for j in range(i + 1, n_orig):
        interactions_train.append((X_train[:, i] * X_train[:, j]).reshape(-1, 1))
        interactions_test.append((X_test[:, i] * X_test[:, j]).reshape(-1, 1))

X_train_eng = np.hstack([X_train] + interactions_train)
X_test_eng  = np.hstack([X_test] + interactions_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_eng, y_train)

preds_eng = rf.predict(X_test_eng)
mse_eng = mean_squared_error(y_test, preds_eng)
r2_eng  = r2_score(y_test, preds_eng)
print(f"  RF + interactions  MSE={mse_eng:.4f}  R²={r2_eng:.4f}")

with tempfile.TemporaryDirectory() as tmpdir:
    model_path   = os.path.join(tmpdir, "rf_model.pkl")
    fnames_path  = os.path.join(tmpdir, "feature_names.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(rf, f)
    with open(fnames_path, "wb") as f:
        pickle.dump(feature_cols, f)

    with mlflow.start_run(run_name="pyfunc_feature_engineering") as run:
        mlflow.log_params({"model_type": "RandomForest", "features": "raw+interactions"})
        mlflow.log_metrics({"mse": mse_eng, "r2": r2_eng})

        mlflow.pyfunc.log_model(
            artifact_path="fe_model",
            python_model=FeatureEngineeringModel(),
            artifacts={
                "model":          model_path,
                "feature_names":  fnames_path,
            },
            input_example=pd.DataFrame(X_test[:3], columns=feature_cols),
        )
        fe_model_uri = f"runs:/{run.info.run_id}/fe_model"
        print(f"  Logged pyfunc model → {fe_model_uri[:50]}...")


# ══════════════════════════════════════════════════════════════════════════════
# C. Load and use pyfunc models — caller only sees a DataFrame in / array out
# ══════════════════════════════════════════════════════════════════════════════
print("\n── C. Loading and using pyfunc models ──")

# Load the preprocessed model
scaled_loaded = mlflow.pyfunc.load_model(scaled_model_uri)
input_df = pd.DataFrame(X_test[:5], columns=feature_cols)
preds_loaded = scaled_loaded.predict(input_df)

print(f"\n  ScaledModel predictions (first 5):")
for i, (pred, actual) in enumerate(zip(preds_loaded, y_test[:5])):
    print(f"    row {i}: predicted={pred:8.4f}  actual={actual:8.4f}")

# Load the feature engineering model
fe_loaded = mlflow.pyfunc.load_model(fe_model_uri)
preds_fe = fe_loaded.predict(input_df)

print(f"\n  FeatureEngineeringModel predictions (first 5):")
for i, (pred, actual) in enumerate(zip(preds_fe, y_test[:5])):
    print(f"    row {i}: predicted={pred:8.4f}  actual={actual:8.4f}")

# Verify: the caller doesn't need to know about scaling or interactions
# The pyfunc contract: DataFrame in → array/Series out
print(f"\n  Both models accept raw features — preprocessing is inside the model")
print(f"  ScaledModel  full test MSE: {mean_squared_error(y_test, scaled_loaded.predict(pd.DataFrame(X_test, columns=feature_cols))):.4f}")
print(f"  FEModel      full test MSE: {mean_squared_error(y_test, fe_loaded.predict(pd.DataFrame(X_test, columns=feature_cols))):.4f}")


print()
print("=" * 64)
print("NEXT: mlflow ui  →  http://localhost:5000")
print("      Open '09_custom_models'. Each run has a pyfunc model artifact.")
print("      The artifact contains your PythonModel class + pkl files.")
print("=" * 64)
print()
print("KEY TAKEAWAYS:")
print("  Subclass mlflow.pyfunc.PythonModel to wrap ANY Python logic as a model")
print("  load_context(context)     — load artifacts once when model is loaded")
print("  predict(context, df)      — run inference (must accept a DataFrame)")
print("  context.artifacts['key']  — path to your logged artifact files")
print("  mlflow.pyfunc.log_model(artifact_path, python_model, artifacts)")
print("  mlflow.pyfunc.load_model(uri)  — load any MLflow model, any flavor")
print("  Preprocessing inside predict() = no silent ordering bugs in production")
