# Module 1: MLflow Tracking — The Foundation

## What You'll Learn

- How to log parameters, metrics, artifacts, and tags
- Step-based metrics for training curves
- Logging matplotlib figures as artifacts
- Nested runs (parent → children) for sweeps

## Why Tracking Matters

Every ML experiment produces four kinds of information that you'll want to
query later:

| Kind | What it is | Example |
|------|-----------|---------|
| **Parameters** | Inputs you control (immutable) | `learning_rate=0.01`, `n_estimators=100` |
| **Metrics** | Results you measure | `accuracy=0.95`, `mse=0.12` |
| **Artifacts** | Files you generate | model files, plots, JSON configs |
| **Tags** | Free-form metadata | `team=ml-eng`, `dataset=v2` |

Without tracking you lose all of this between runs. MLflow stores it all in
a structured, queryable format backed by a local `mlruns/` directory (or a
remote server in production).

## Core Concepts

### Run

A **run** is one execution of your training code. It has a unique run ID,
a name, a start/end time, a status, and everything logged during that execution.

```python
with mlflow.start_run(run_name="my_run") as run:
    print(run.info.run_id)   # e.g. "abc123def456..."
    mlflow.log_param("lr", 0.01)
    mlflow.log_metric("accuracy", 0.95)
```

The `with` block ends the run cleanly even if an exception is raised.

### Parameters vs Metrics

```python
# Parameters: set before training, immutable once logged
mlflow.log_param("model_type", "RandomForest")
mlflow.log_params({"n_estimators": 100, "max_depth": 8})

# Metrics: measured after/during training, can be updated, support steps
mlflow.log_metric("accuracy", 0.92)
mlflow.log_metric("loss", 0.14, step=10)    # step = epoch or iteration
mlflow.log_metrics({"f1": 0.91, "auc": 0.97})
```

### Step-Based Metrics

Logging a metric with `step=` creates a time-series chart in the UI — essential
for tracking training curves in deep learning, or seeing how performance
changes as you add more data.

```python
for epoch in range(100):
    train_loss = ...
    val_loss   = ...
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("val_loss",   val_loss,   step=epoch)
```

### Artifacts

Any file can be an artifact. MLflow stores it in the run's artifact folder:

```python
mlflow.log_artifact("confusion_matrix.png")           # file
mlflow.log_artifact("confusion_matrix.png", "plots")  # into subfolder
mlflow.log_text("y = 2.5 * X + 1", "equation.txt")   # string → file
mlflow.log_dict({"r2": 0.95}, "summary.json")         # dict → JSON
mlflow.log_figure(fig, "residuals.png")               # matplotlib figure
```

### Nested Runs

A parent run can own multiple child runs. Use this when you want one
high-level run to represent an experiment (e.g., a hyperparameter sweep),
with each configuration as an independently inspectable child run.

```python
with mlflow.start_run(run_name="sweep") as parent:
    for alpha in [0.1, 1.0, 10.0]:
        with mlflow.start_run(run_name=f"alpha_{alpha}", nested=True):
            # each child has its own params, metrics, and artifacts
            mlflow.log_param("alpha", alpha)
```

## API Reference

```python
import mlflow

# Run lifecycle
mlflow.start_run(run_name="name")       # start (use as context manager)
mlflow.active_run()                     # get current run
mlflow.end_run()                        # end (automatic with context manager)

# Logging
mlflow.log_param("name", value)
mlflow.log_params({"a": 1, "b": 2})
mlflow.log_metric("name", value)
mlflow.log_metric("name", value, step=10)
mlflow.log_metrics({"acc": 0.9, "loss": 0.1})
mlflow.log_artifact("path/to/file.ext")
mlflow.log_artifact("file.ext", "subfolder")
mlflow.log_text("content", "file.txt")
mlflow.log_dict({"k": "v"}, "file.json")
mlflow.log_figure(fig, "plot.png")
mlflow.sklearn.log_model(model, name="model", input_example=X[:3])

# Tags
mlflow.set_tag("key", "value")
mlflow.set_tags({"team": "ml", "env": "dev"})
```

## Run It

```bash
python 01-tracking/tracking.py

# Then explore:
mlflow ui
# http://localhost:5000
```

## What to Explore in the UI

1. **Run A → Metrics tab**: click `test_mse_curve` to see the step-indexed chart
2. **Run A → Artifacts tab**: open `plots/` to see the diagnostic PNG
3. **Run B**: expand the nested child runs and compare MSE across alphas

## Exercises

1. Add `max_error` as a metric to Run A
2. Log a second plot (e.g., a histogram of residuals)
3. Change the noise level to 3.0 — observe how R² and MSE change
4. Add your own tag (e.g., `"your_name"`)
5. In the nested sweep, add a third metric: `mae`

## Key Takeaways

- `log_param()` for inputs, `log_metric()` for results, `log_artifact()` for files
- `step=` parameter on `log_metric()` creates time-series charts
- `log_model()` saves the trained model with schema auto-detected from `input_example`
- `nested=True` in `start_run()` creates child runs under the current active run
- Everything is stored in `mlruns/` — browse it with `mlflow ui`

## Next

**[Module 2: Experiments →](../02-experiments/README.md)**
