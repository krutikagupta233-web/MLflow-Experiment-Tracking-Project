# Module 8: MLflow Evaluate — Systematic Model Assessment

## What You'll Learn

- How `mlflow.evaluate()` works and why it matters
- Built-in metrics for regression and classification
- How to define a custom metric with `make_metric()`
- Baseline comparison — is your model better than trivial?
- Why evaluation should be a separate run from training

## Why Evaluation Matters

Manually computing metrics and logging them is fine for one model. But when
you're comparing many models, you want:

1. **Consistent metrics** — same computation every time
2. **Automatic artifact logging** — prediction tables, error plots
3. **Baseline comparison** — how much better than "always predict mean"?
4. **Reproducibility** — evaluation code is tracked like training code

`mlflow.evaluate()` gives you all of this with one call.

## How It Works

```python
result = mlflow.evaluate(
    model="runs:/<RUN_ID>/model",   # or a Python callable
    data=eval_df,                    # DataFrame with features + target column
    targets="target",               # name of target column
    model_type="regressor",         # or "classifier"
)

print(result.metrics["root_mean_squared_error"])
print(result.metrics["r2_score"])
```

MLflow:
1. Loads the model
2. Runs inference on `eval_df` (minus the `targets` column)
3. Computes all standard metrics
4. Logs metrics + artifacts to the active run
5. Returns a `EvaluationResult` object with `.metrics` and `.artifacts`

## Built-in Metrics

### Regression (`model_type="regressor"`)

| Metric key | Description |
|-----------|-------------|
| `mean_absolute_error` | Average absolute error |
| `mean_squared_error` | Average squared error |
| `root_mean_squared_error` | Square root of MSE |
| `r2_score` | Coefficient of determination |
| `mean_absolute_percentage_error` | MAPE (%) |
| `max_error` | Worst-case absolute error |

### Classification (`model_type="classifier"`)

| Metric key | Description |
|-----------|-------------|
| `accuracy_score` | Fraction correct |
| `f1_score` | Harmonic mean of precision and recall |
| `precision_score` | TP / (TP + FP) |
| `recall_score` | TP / (TP + FN) |
| `roc_auc` | Area under ROC curve |
| `log_loss` | Negative log-likelihood |

## Custom Metrics

```python
from mlflow.metrics import make_metric, MetricValue

def my_fn(predictions, targets, metrics, sample_weights=None):
    import numpy as np
    val = float(np.mean(np.abs(targets.values - predictions.values)))
    return MetricValue(aggregate_results={"mean": val})

my_metric = make_metric(
    eval_fn=my_fn,
    name="my_metric",
    greater_is_better=False,
)

result = mlflow.evaluate(..., extra_metrics=[my_metric])
print(result.metrics["my_metric/mean"])
```

`MetricValue` can include:
- `aggregate_results` — dict of summary stats (always required)
- `scores` — per-row list (optional, for row-level inspection)

## Baseline Comparison

Pass a Python callable as `model` to evaluate a naive baseline:

```python
def predict_mean(model_input):
    return np.full(len(model_input), y_train.mean())

with mlflow.start_run(run_name="baseline"):
    result = mlflow.evaluate(
        model=predict_mean,
        data=eval_df,
        targets="target",
        model_type="regressor",
    )
```

Compare the baseline RMSE to your model's RMSE — the gap tells you how much
value your model actually adds.

## Best Practice: Separate Training and Evaluation Runs

```
train_LinearRegression  (run)    → logs training params + model artifact
eval_LinearRegression   (run)    → logs evaluation metrics + prediction table
```

This separation means:
- You can re-evaluate after fixing a bug without retraining
- Evaluation parameters (eval dataset version, metric set) are versioned
- The UI shows training vs evaluation side by side

## API Reference

```python
import mlflow
from mlflow.metrics import make_metric, MetricValue

# Basic evaluation
result = mlflow.evaluate(
    model="runs:/<RUN_ID>/model",    # URI or callable
    data=eval_df,                     # DataFrame with target column
    targets="target",                # target column name
    model_type="regressor",          # "regressor" or "classifier"
    evaluators="default",            # or list of evaluator names
    extra_metrics=[custom_metric],   # list of make_metric() objects
)

# Access results
result.metrics          # dict of all computed metrics
result.artifacts        # dict of artifact paths

# Custom metric
def fn(predictions, targets, metrics, sample_weights=None):
    return MetricValue(aggregate_results={"mean": float(...)})

metric = make_metric(eval_fn=fn, name="my_metric", greater_is_better=True)
```

## Run It

```bash
python 08-evaluation/evaluation.py
mlflow ui
```

## What to Explore in the UI

1. Open `08_evaluation` — two sets of runs: `train_*` and `eval_*`
2. Select all `eval_*` runs → Compare — instantly see which model wins
3. Click an `eval_*` run → Artifacts → `eval_results_table.json` — per-row predictions
4. Compare `eval_baseline_mean` vs best model RMSE

## Exercises

1. Add `mlflow.evaluate()` for a **classifier** — use `RandomForestClassifier` with a binary target
2. Write a custom metric: `within_5pct` (threshold 5% instead of 10%)
3. Add `model_type="regressor"` with a quantile error metric
4. Compare a model trained on 50% of the data vs 100% — does more data help?
5. Use `baseline_model` parameter instead of a callable

## Key Takeaways

- `mlflow.evaluate()` standardises model assessment across all experiments
- Built-in metrics cover all standard regression and classification measures
- `make_metric()` lets you define domain-specific custom metrics
- Evaluate callables directly for baselines — no run/artifact required
- Keep training and evaluation in separate runs for clean auditability

## Next

**[Module 9: Custom Models →](../09-custom-models/README.md)**
