# Module 10: Hyperparameter Tuning — Nested Runs

## What You'll Learn

- How nested runs model a tuning experiment (parent = sweep, children = trials)
- Grid search, 2D grid search, and random search with MLflow
- Programmatically finding the best child run across all sweeps
- Registering and promoting the winner automatically
- When to use grid vs random search

## The Pattern

```
Parent run: sweep_ridge_alpha
  ├── Child: ridge_alpha_0.001   params + metrics
  ├── Child: ridge_alpha_0.01
  ├── Child: ridge_alpha_0.1     ← best MSE
  └── Child: ridge_alpha_1.0
      ↑
Parent logs: best_alpha, best_mse  (summary of the sweep)
```

Every trial is independently inspectable. The parent gives you a summary.
The hierarchy shows up cleanly in the UI.

## Grid vs Random Search

| | Grid Search | Random Search |
|--|------------|---------------|
| Coverage | Exhaustive | Probabilistic |
| 10 params × 5 values | 10^5 = 100,000 trials | 100 trials |
| Best for | Small grids, known ranges | Large/unknown spaces |
| MLflow pattern | Same nested run pattern | Same nested run pattern |

For most real tuning jobs with ≥3 hyperparameters, random search is
more efficient than grid search.

## Nested Run Pattern

```python
with mlflow.start_run(run_name="sweep") as parent:
    mlflow.set_tags({"sweep_type": "grid", "model": "Ridge"})

    best_mse = float("inf")
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        with mlflow.start_run(run_name=f"alpha_{alpha}", nested=True):
            # each child: train, evaluate, log
            mlflow.log_param("alpha", alpha)
            mlflow.log_metric("mse", mse)
            if mse < best_mse:
                best_mse = mse

    # Log summary to parent
    mlflow.log_metric("best_mse", best_mse)
```

## Finding the Best Run Programmatically

```python
all_runs = mlflow.search_runs(
    experiment_names=["10_hyperparameter_tuning"],
    filter_string='status = "FINISHED"',
    order_by=["metrics.mse ASC"],
)

best_run_id = all_runs.iloc[0]["run_id"]
best_mse    = all_runs.iloc[0]["metrics.mse"]
```

Then register:
```python
registered = mlflow.register_model(f"runs:/{best_run_id}/model", "MyModel")
client.set_registered_model_alias("MyModel", "champion", registered.version)
```

## API Reference

```python
# Nested runs
with mlflow.start_run(run_name="sweep") as parent:
    for param in grid:
        with mlflow.start_run(run_name=f"trial_{param}", nested=True):
            mlflow.log_param("param", param)
            mlflow.log_metric("metric", value)

# Register and promote after the sweep
registered = mlflow.register_model(f"runs:/{best_run_id}/model", "ModelName")
client.set_registered_model_alias("ModelName", "champion", registered.version)
```

## Run It

```bash
python 10-hyperparameter/hyperparameter.py
mlflow ui
```

## What to Explore in the UI

1. Open `10_hyperparameter_tuning`
2. Click `sweep_ridge_alpha` → expand nested child runs
3. Click any child run to see its full details
4. Select all child runs from one sweep → Compare → parallel coordinates plot
5. Models tab → `TunedModel` — see the champion version

## Exercises

1. Add a sweep for `Lasso` (alpha grid)
2. Add a third dimension to the RF grid: `min_samples_split`
3. Implement early stopping: break if 5 consecutive trials don't improve
4. Write a function `best_child_run(parent_run_id)` that returns the best child's run_id
5. After tuning, use `mlflow.evaluate()` (Module 8) on the champion to generate a full metric report

## When to Use What

| Scenario | Approach |
|---------|---------|
| 1 param, narrow range | Grid with 5-10 values |
| 2-3 params, known ranges | Grid if <100 combos, else random |
| 3+ params, wide ranges | Random search, 50-200 trials |
| Complex interdependencies | Bayesian optimisation (Optuna + MLflow) |

## Key Takeaways

- `nested=True` in `start_run()` creates child runs under the active parent
- Parent run = the sweep summary; child runs = individual trials
- `search_runs(order_by=["metrics.mse ASC"])` finds the best trial
- `register_model()` + `set_registered_model_alias()` promotes the winner
- Random search finds good configs faster than grid for large parameter spaces
- The UI hierarchy makes it easy to compare trials within and across sweeps

---

## Curriculum Complete

You've covered all 10 modules:

| Module | Concept |
|--------|---------|
| 1 — Tracking | log_param, log_metric, step curves, nested runs |
| 2 — Experiments | grouping, search_runs, filter_string |
| 3 — Autologging | zero-effort tracking, GridSearchCV |
| 4 — Registry | versions, aliases, lifecycle |
| 5 — Serving | REST API, all input formats, direct loading |
| 6 — Projects | reproducible packaging, mlflow run |
| 7 — Pipeline | end-to-end template |
| 8 — Evaluation | mlflow.evaluate(), custom metrics, baseline |
| 9 — Custom Models | pyfunc, preprocessing inside the model |
| 10 — Hyperparameter | nested runs, grid search, random search, promotion |

**Next steps:**
- MLflow on a remote tracking server (Postgres + S3)
- Bayesian optimisation with Optuna + MLflow
- MLflow with PyTorch / TensorFlow training loops
- Deploying MLflow models to Kubernetes or SageMaker
- Drift monitoring with Evidently + MLflow logging
