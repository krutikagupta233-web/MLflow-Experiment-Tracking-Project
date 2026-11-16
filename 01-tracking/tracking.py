"""
================================================================================
MODULE 1: MLflow Tracking — The Foundation
================================================================================
Core idea: Every ML run should record what went in (params), what came out
(metrics), and what was produced (artifacts). MLflow stores all of it in a
queryable, UI-browsable format.

This module covers:
  A. Basic params, metrics, artifacts, tags
  B. Step-based metrics (training curves)
  C. Matplotlib figures as artifacts
  D. Nested runs (parent → children)
================================================================================
"""

import os
import tempfile

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── Data ──────────────────────────────────────────────────────────────────────
np.random.seed(42)
X = np.random.rand(200, 1) * 10
y = 2.5 * X.squeeze() + 1 + np.random.randn(200) * 0.8

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("01_basic_tracking")


# ══════════════════════════════════════════════════════════════════════════════
# RUN A: Full logging demo — params, metrics, step curves, artifacts, tags
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Run A: Full tracking demo ──")

with mlflow.start_run(run_name="linear_regression_full_demo") as run:
    print(f"  Run ID: {run.info.run_id}")

    # 1. PARAMS — inputs/config, immutable once logged
    mlflow.log_params({
        "model_type":   "LinearRegression",
        "test_size":    0.2,
        "random_state": 42,
        "n_samples":    len(X),
        "noise_level":  0.8,
    })

    # 2. TRAIN
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # 3. METRICS — numeric results
    mlflow.log_metrics({
        "mse":         mse,
        "rmse":        rmse,
        "mae":         mae,
        "r2":          r2,
        "coefficient": float(model.coef_[0]),
        "intercept":   float(model.intercept_),
    })

    # 4. STEP-BASED METRICS — simulate a learning curve (key for deep learning)
    #    We retrain on growing subsets to show how metrics evolve over "steps"
    for step, subset_size in enumerate(range(20, len(X_train) + 1, 10)):
        m_sub = LinearRegression().fit(X_train[:subset_size], y_train[:subset_size])
        train_mse = mean_squared_error(y_train[:subset_size], m_sub.predict(X_train[:subset_size]))
        test_mse  = mean_squared_error(y_test, m_sub.predict(X_test))
        mlflow.log_metric("train_mse_curve", train_mse, step=step)
        mlflow.log_metric("test_mse_curve",  test_mse,  step=step)

    # 5. TAGS — free-form searchable metadata
    mlflow.set_tags({
        "team":      "ml-engineering",
        "dataset":   "synthetic_linear",
        "framework": "sklearn",
    })

    # 6. LOG THE MODEL with input signature auto-detected from example
    mlflow.sklearn.log_model(
        sk_model=model,
        name="linear_model",
        input_example=X_test[:3],
    )

    # 7. LOG TEXT AND DICT ARTIFACTS
    equation = f"y = {model.coef_[0]:.4f} * X + {model.intercept_:.4f}"
    mlflow.log_text(equation, "model_equation.txt")
    mlflow.log_dict({
        "coefficient":    float(model.coef_[0]),
        "intercept":      float(model.intercept_),
        "expected_slope": 2.5,
        "r2":             float(r2),
    }, "model_summary.json")

    # 8. LOG A MATPLOTLIB FIGURE AS ARTIFACT
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.scatter(y_test, y_pred, alpha=0.6, color="steelblue", edgecolors="white", linewidths=0.4)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax1.plot(lims, lims, "r--", lw=1.5)
    ax1.set_xlabel("Actual"); ax1.set_ylabel("Predicted")
    ax1.set_title(f"Predicted vs Actual  (R²={r2:.3f})")

    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, color="coral", edgecolors="white", linewidths=0.4)
    ax2.axhline(0, color="black", lw=1.5, linestyle="--")
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Residual")
    ax2.set_title("Residual Plot")

    plt.suptitle("Linear Regression Diagnostics", fontsize=13)
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        plt.savefig(f.name, dpi=120, bbox_inches="tight")
        mlflow.log_artifact(f.name, "plots")
        tmp_path = f.name
    plt.close()
    os.unlink(tmp_path)

    print(f"  R²={r2:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}  Equation: {equation}")


# ══════════════════════════════════════════════════════════════════════════════
# RUN B: Nested runs — parent run wraps multiple child runs
# Best for: hyperparameter sweeps, CV folds, ensemble components
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Run B: Nested runs (alpha sweep) ──")

alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

with mlflow.start_run(run_name="ridge_alpha_sweep") as parent_run:
    mlflow.set_tag("run_type", "hyperparameter_sweep")
    mlflow.log_param("param_grid_alphas", str(alphas))

    best_mse   = float("inf")
    best_alpha = None

    for alpha in alphas:
        with mlflow.start_run(run_name=f"ridge_alpha_{alpha}", nested=True):
            m = Ridge(alpha=alpha)
            m.fit(X_train, y_train)
            preds = m.predict(X_test)
            child_mse = mean_squared_error(y_test, preds)
            child_r2  = r2_score(y_test, preds)

            mlflow.log_params({"alpha": alpha, "model_type": "Ridge"})
            mlflow.log_metrics({"mse": child_mse, "r2": child_r2})
            mlflow.sklearn.log_model(m, "model")

            print(f"    alpha={alpha:>6}  MSE={child_mse:.4f}  R²={child_r2:.4f}")

            if child_mse < best_mse:
                best_mse   = child_mse
                best_alpha = alpha

    # Log winner summary back to the parent
    mlflow.log_metrics({"best_mse": best_mse})
    mlflow.log_param("best_alpha", best_alpha)
    print(f"  Best: alpha={best_alpha}  MSE={best_mse:.4f}")


print()
print("=" * 62)
print("NEXT: mlflow ui  →  http://localhost:5000")
print("      Open '01_basic_tracking'.")
print("      Run A → Artifacts tab → plots folder for the diagnostic chart.")
print("      Run A → Metrics tab → click 'test_mse_curve' to see the step chart.")
print("      Run B → expand the nested child runs to compare alphas.")
print("=" * 62)
print()
print("KEY TAKEAWAYS:")
print("  log_params()         — configuration inputs (immutable)")
print("  log_metrics()        — numeric results")
print("  log_metric(..., step) — step-indexed series (training curves)")
print("  log_model()          — save trained model with signature")
print("  log_text()           — save a string as a file artifact")
print("  log_dict()           — save a dict as JSON artifact")
print("  log_artifact()       — save any file")
print("  set_tags()           — searchable free-form metadata")
print("  nested=True          — child runs under a parent run")
