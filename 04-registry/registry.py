"""
================================================================================
MODULE 4: MLflow Model Registry — Version, Alias, and Lifecycle
================================================================================
The Model Registry is a centralized store for trained models. It lets you:
  - Version models (each log_model call creates a new version)
  - Assign named aliases ("champion", "challenger") to specific versions
  - Inspect metadata, add descriptions, and track which run produced each version
  - Load any version by alias, version number, or stage

Modern MLflow (2.x) prefers ALIASES over STAGES for production routing.
Stages (Staging / Production / Archived) are deprecated but still work —
you'll see them in a lot of existing code, so we cover both.
================================================================================
"""

import time

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── Data ──────────────────────────────────────────────────────────────────────
np.random.seed(42)
X = np.random.rand(300, 3) * 10
y = 2.5 * X[:, 0] + 1.5 * X[:, 1] + 3.0 * X[:, 2] + 1 + np.random.randn(300) * 0.5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("04_model_registry")
REGISTERED_NAME = "RegressionModel"

client = MlflowClient()


# ══════════════════════════════════════════════════════════════════════════════
# 1. REGISTER MODELS — passing registered_model_name to log_model() is enough
#    Each call creates a new version automatically
# ══════════════════════════════════════════════════════════════════════════════
print("── Registering model versions ──\n")

run_ids = {}

# Version: baseline LinearRegression
with mlflow.start_run(run_name="v_baseline_linear") as run:
    m = LinearRegression()
    m.fit(X_train, y_train)
    mse = mean_squared_error(y_test, m.predict(X_test))

    mlflow.log_params({"model_type": "LinearRegression"})
    mlflow.log_metrics({"mse": mse, "r2": r2_score(y_test, m.predict(X_test))})
    mlflow.sklearn.log_model(
        sk_model=m,
        name="model",
        registered_model_name=REGISTERED_NAME,  # triggers registration
        input_example=X_test[:3],
    )
    run_ids["baseline"] = run.info.run_id
    print(f"  Baseline (LinearRegression) — MSE={mse:.4f}")

# Version: Ridge with alpha search
best_run_mse = float("inf")
best_alpha   = None

for alpha in [0.01, 0.1, 1.0, 5.0, 10.0]:
    with mlflow.start_run(run_name=f"ridge_alpha_{alpha}") as run:
        m = Ridge(alpha=alpha)
        m.fit(X_train, y_train)
        mse = mean_squared_error(y_test, m.predict(X_test))
        r2  = r2_score(y_test, m.predict(X_test))

        mlflow.log_params({"model_type": "Ridge", "alpha": alpha})
        mlflow.log_metrics({"mse": mse, "r2": r2})
        mlflow.sklearn.log_model(
            sk_model=m,
            name="model",
            registered_model_name=REGISTERED_NAME,
            input_example=X_test[:3],
        )
        if mse < best_run_mse:
            best_run_mse = mse
            best_alpha   = alpha
            run_ids["champion_candidate"] = run.info.run_id

        print(f"  Ridge alpha={alpha:5}  MSE={mse:.4f}  R²={r2:.4f}")

# Give MLflow a moment to finish background registration writes
time.sleep(1)

# ══════════════════════════════════════════════════════════════════════════════
# 2. FIND VERSIONS BY RUN ID (never hardcode version numbers)
#    Version numbers accumulate across script runs — always look them up
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n── Registry state ──")

all_versions = client.search_model_versions(f"name='{REGISTERED_NAME}'")

# Build a map: run_id → version
run_to_version = {mv.run_id: mv.version for mv in all_versions}

baseline_version  = run_to_version.get(run_ids["baseline"])
champion_version  = run_to_version.get(run_ids["champion_candidate"])

for mv in sorted(all_versions, key=lambda v: int(v.version)):
    tag = ""
    if mv.run_id == run_ids["baseline"]:
        tag = "  ← baseline"
    elif mv.run_id == run_ids["champion_candidate"]:
        tag = "  ← best Ridge"
    print(f"  v{mv.version:>2}  run={mv.run_id[:12]}...  stage={mv.current_stage:<8}{tag}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. ALIASES (modern approach — preferred over stages in MLflow 2.x)
#    An alias is a human-readable pointer: "champion", "challenger", "shadow"
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n── Setting aliases ──")

client.set_registered_model_alias(REGISTERED_NAME, "champion",   champion_version)
client.set_registered_model_alias(REGISTERED_NAME, "challenger", baseline_version)
print(f"  champion   → v{champion_version}")
print(f"  challenger → v{baseline_version}")

# Add descriptions
client.update_model_version(
    name=REGISTERED_NAME,
    version=champion_version,
    description=f"Best Ridge model  alpha={best_alpha}  MSE={best_run_mse:.4f}",
)
client.set_model_version_tag(REGISTERED_NAME, champion_version, "approved_by", "pipeline")
client.update_registered_model(REGISTERED_NAME, description="Regression model — synthetic 3-feature dataset")


# ══════════════════════════════════════════════════════════════════════════════
# 4. STAGE-BASED API (deprecated but widely used in legacy code)
#    Shown here so you can read existing code — use aliases for new work
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n── Stage transitions (legacy API) ──")

# Promote champion to Production via stage
client.transition_model_version_stage(REGISTERED_NAME, champion_version,  "Production")
print(f"  v{champion_version} → Production  (deprecated API — prefer aliases)")

# Archive old versions
for mv in client.search_model_versions(f"name='{REGISTERED_NAME}'"):
    if mv.version != champion_version and mv.current_stage not in ("Archived", "None"):
        client.transition_model_version_stage(REGISTERED_NAME, mv.version, "Archived")
        print(f"  v{mv.version} → Archived")


# ══════════════════════════════════════════════════════════════════════════════
# 5. LOAD MODELS — three URI patterns
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n── Loading from registry ──")

# By alias (preferred)
champion_model = mlflow.sklearn.load_model(f"models:/{REGISTERED_NAME}@champion")
preds = champion_model.predict(X_test)
print(f"  @champion alias  MSE={mean_squared_error(y_test, preds):.4f}")

# By explicit version number
v_model = mlflow.sklearn.load_model(f"models:/{REGISTERED_NAME}/{champion_version}")
preds_v = v_model.predict(X_test)
print(f"  Version {champion_version:>2} explicit  MSE={mean_squared_error(y_test, preds_v):.4f}")

# By stage (deprecated, but you'll see this in legacy code)
prod_model = mlflow.sklearn.load_model(f"models:/{REGISTERED_NAME}/Production")
preds_p = prod_model.predict(X_test)
print(f"  /Production stage MSE={mean_squared_error(y_test, preds_p):.4f}")

# Generic pyfunc loader (works for any flavor — sklearn, pytorch, etc.)
generic = mlflow.pyfunc.load_model(f"models:/{REGISTERED_NAME}@champion")
import pandas as pd
preds_g = generic.predict(pd.DataFrame(X_test, columns=["f0", "f1", "f2"]))
print(f"  pyfunc generic   MSE={mean_squared_error(y_test, preds_g):.4f}")


print()
print("=" * 62)
print("NEXT: mlflow ui  →  http://localhost:5000")
print("      Click 'Models' tab to see 'RegressionModel'.")
print("      Inspect versions, aliases, and descriptions.")
print("=" * 62)
print()
print("KEY TAKEAWAYS:")
print("  registered_model_name in log_model() → auto-registers a new version")
print("  Never hardcode version numbers — look them up by run_id")
print("  set_registered_model_alias()  → modern way to tag versions (preferred)")
print("  transition_model_version_stage() → legacy stages (deprecated but works)")
print("  Load by alias:   models:/Name@champion   (preferred)")
print("  Load by version: models:/Name/3           (explicit)")
print("  Load by stage:   models:/Name/Production  (deprecated)")
print("  mlflow.pyfunc.load_model() → works for any flavor")
