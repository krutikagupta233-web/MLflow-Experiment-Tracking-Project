# Module 2: MLflow Experiments — Organise, Search, Compare

## What You'll Learn

- How experiments group runs (and why that matters)
- Creating and selecting experiments programmatically
- Searching runs with `filter_string` and `order_by`
- Using `MlflowClient` for low-level run inspection
- Comparing runs side-by-side in the UI

## Core Concepts

### Experiment

An **experiment** is a named container for related runs. Every run belongs
to exactly one experiment. Without explicit selection, runs land in the
`Default` experiment.

```python
# Create (raises an exception if name already exists)
mlflow.create_experiment("my_experiment", tags={"purpose": "testing"})

# Idiomatic pattern — create or reuse
try:
    exp_id = mlflow.create_experiment("my_experiment")
except mlflow.exceptions.MlflowException:
    exp_id = mlflow.get_experiment_by_name("my_experiment").experiment_id

# Point all subsequent runs here
mlflow.set_experiment("my_experiment")
```

### Searching Runs

`mlflow.search_runs()` returns a **pandas DataFrame** where:
- Columns prefixed `metrics.` hold metric values
- Columns prefixed `params.` hold parameter values
- Columns prefixed `tags.` hold tag values

```python
df = mlflow.search_runs(experiment_names=["my_experiment"])

# Filter: SQL-like filter_string
df = mlflow.search_runs(
    experiment_names=["my_experiment"],
    filter_string='metrics.mse < 0.05 AND tags.env = "prod"',
    order_by=["metrics.mse ASC"],
    max_results=20,
)

# Standard pandas operations
best = df.loc[df["metrics.mse"].idxmin()]
print(best["tags.mlflow.runName"], best["metrics.mse"])
```

### Filter String Syntax

```
metrics.accuracy > 0.9
metrics.mse < 0.1 AND metrics.r2 > 0.95
params.model_type = "Ridge"
tags.team = "ml-eng"
tags.model_family = "RandomForest" OR tags.model_family = "GradientBoost"
status = "FINISHED"
```

### MlflowClient

`MlflowClient` gives low-level access to individual runs:

```python
from mlflow.tracking import MlflowClient
client = MlflowClient()

run = client.get_run(run_id)
print(run.data.params)     # dict
print(run.data.metrics)    # dict
print(run.data.tags)       # dict
print(run.info.status)     # RUNNING / FINISHED / FAILED
```

## API Reference

```python
# Create / select
mlflow.create_experiment(name, tags={"key": "value"})
mlflow.set_experiment("name")
mlflow.get_experiment_by_name("name")   # → Experiment object

# Search
df = mlflow.search_runs(
    experiment_ids=["1", "2"],        # by ID
    experiment_names=["exp_name"],    # or by name
    filter_string="...",              # SQL-like filter
    order_by=["metrics.mse ASC"],     # sort
    max_results=100,
)

# Client
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.get_run(run_id)
client.search_experiments()
```

## Run It

```bash
python 02-experiments/experiments.py
mlflow ui
```

## What to Explore in the UI

1. Click `02_model_comparison` in the left sidebar
2. Select all runs → click **Compare** at the top
3. The parallel coordinates chart shows all metrics — drag axes to reorder
4. Sort the run table by MSE or R² to rank models instantly
5. Click any run to see its params, metrics, and artifacts

## Exercises

1. Add a `cv_mae` metric (5-fold CV on MAE)
2. Search for runs where `cv_r2 > 0.99`
3. Find the run with the highest `cv_r2` using `search_runs` + pandas
4. Add a tag `"model_family"` and filter by it
5. Use `client.get_run()` to print the artifact URI of the best run

## Key Takeaways

- `create_experiment()` organizes runs; wrap in try/except for idempotency
- `search_runs()` returns a DataFrame — use pandas to rank, filter, analyze
- `filter_string` is SQL-like; supports `AND`, `OR`, comparisons
- `MlflowClient` gives low-level access to any run or experiment object
- The UI **Compare** view is the fastest way to visually rank models

## Next

**[Module 3: Autologging →](../03-autologging/README.md)**
