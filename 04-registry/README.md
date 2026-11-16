# Module 4: MLflow Model Registry — Version, Alias, Lifecycle

## What You'll Learn

- How to register a model and create versions automatically
- Why you should never hardcode version numbers
- Aliases vs stages — what changed and which to use
- How to load a model by alias, version, and stage
- Metadata: descriptions, tags, run-to-version mapping

## Why the Registry Matters

The Registry decouples *training* from *deployment*. Your serving code loads
a model by a stable name like `models:/RegressionModel@champion` — and you
can swap what "champion" points to without touching the serving code.

```
Training pipeline           Serving code
──────────────              ─────────────────────────────────
trains 10 models     →      model = mlflow.sklearn.load_model(
promotes best one               "models:/RegressionModel@champion"
                            )
```

## Model Lifecycle

```
Registered (version N)
       │
    Aliases (modern)         Stages (deprecated)
   ────────────────         ───────────────────
   @champion                 → Production
   @challenger               → Staging
   @shadow                   → (None)
                             → Archived
```

## Aliases vs Stages

| | Aliases | Stages |
|--|---------|--------|
| API | `set_registered_model_alias()` | `transition_model_version_stage()` |
| Status | **Current (preferred)** | Deprecated in MLflow 2.x |
| Names | Any string you choose | Fixed: None/Staging/Production/Archived |
| Multiple per version | Yes | No |
| Multiple per model | Yes | One per stage name |

Use **aliases** for all new work. You'll still see stages in existing code
(including legacy MLflow examples) — that's why `registry.py` shows both.

## The Version Number Trap

**Never hardcode version numbers.** Every time you run `registry.py`, new
versions are created. If you hardcode `version=2`, it works the first time
but points to the wrong model on subsequent runs.

Instead, look up the version by `run_id`:

```python
# WRONG — breaks on re-runs
client.set_registered_model_alias("MyModel", "champion", version=2)

# CORRECT — always points to the model you actually trained
all_versions = client.search_model_versions("name='MyModel'")
version_map  = {mv.run_id: mv.version for mv in all_versions}
my_version   = version_map[my_run_id]
client.set_registered_model_alias("MyModel", "champion", my_version)
```

## API Reference

```python
from mlflow.tracking import MlflowClient
client = MlflowClient()

# Register during training (most common)
mlflow.sklearn.log_model(model, "model", registered_model_name="MyModel")

# Register an existing run's model
mlflow.register_model(f"runs:/{run_id}/model", "MyModel")

# Aliases
client.set_registered_model_alias("MyModel", "champion", version)
client.delete_registered_model_alias("MyModel", "champion")

# Stages (deprecated — still works)
client.transition_model_version_stage("MyModel", version, "Production")

# Metadata
client.update_model_version("MyModel", version, description="Best model")
client.set_model_version_tag("MyModel", version, "approved_by", "alice")
client.update_registered_model("MyModel", description="Main regressor")

# Load
mlflow.sklearn.load_model("models:/MyModel@champion")     # by alias
mlflow.sklearn.load_model("models:/MyModel/3")            # by version
mlflow.sklearn.load_model("models:/MyModel/Production")   # by stage (deprecated)
mlflow.pyfunc.load_model("models:/MyModel@champion")      # any flavor

# Inspect
for mv in client.search_model_versions("name='MyModel'"):
    print(mv.version, mv.current_stage, mv.aliases, mv.run_id)
```

## Run It

```bash
python 04-registry/registry.py
mlflow ui
```

## What to Explore in the UI

1. Click the **Models** tab in the top nav
2. Select `RegressionModel` — see all versions with their stages and aliases
3. Click a version — see the description, tags, and which run produced it
4. Click the run link — jump directly to the training run

## Exercises

1. Register a new version by training a `GradientBoostingRegressor`
2. Assign it an alias `"experiment"` and load it by that alias
3. Move the current champion's alias to your new version
4. Add a custom tag `"dataset_version"` to the champion version
5. Use `mlflow.pyfunc.load_model()` and make predictions — what's different?

## Key Takeaways

- `registered_model_name` in `log_model()` auto-creates a new version
- **Never hardcode version numbers** — look them up via `run_id`
- Aliases are the modern way to identify versions for production
- Stage transitions still work but are deprecated — use aliases for new code
- The URI format `models:/Name@alias` decouples training from serving

## Next

**[Module 5: Serving →](../05-serving/README.md)**
