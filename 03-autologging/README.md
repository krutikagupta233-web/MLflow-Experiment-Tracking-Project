# Module 3: MLflow Autologging — Zero-Effort Tracking

## What You'll Learn

- How autologging captures everything with one line of code
- What gets logged automatically vs what you add manually
- Autologging with `GridSearchCV` (per-trial child runs)
- Fine-grained options: `log_models`, `log_input_examples`, `max_tuning_runs`
- Supported frameworks

## Why Autologging Matters

Writing `log_param` / `log_metric` calls for every hyperparameter is tedious
and easy to forget. Autologging fires on `model.fit()` and captures everything
automatically:

```python
mlflow.sklearn.autolog()   # one line before training

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)   # autolog fires here — everything is captured
```

After `model.fit()`, the run has:
- All constructor params (`alpha`, `fit_intercept`, `max_iter`, ...)
- Training metrics (`training_r2_score`, `training_mean_squared_error`, ...)
- The trained model (as a loadable artifact)
- Input example (a few rows of training data)
- Model signature (input/output schema for serving)

**All without a single manual log call.**

## What Autolog Captures

### Parameters — every constructor argument
```
params.alpha = 1.0
params.fit_intercept = True
params.normalize = False      # (sklearn version dependent)
params.n_jobs = None
...
```

### Metrics — training scores computed by sklearn
```
metrics.training_score = 0.994
metrics.training_mean_squared_error = 0.083
metrics.training_r2_score = 0.994
metrics.training_mean_absolute_error = 0.224
```

### Model — the trained artifact
Saved as a loadable MLflow model with flavor metadata.

### Input Example + Signature
- A sample from `X_train` so the model documents its own input shape
- The signature tells downstream tools (serving, UI) what inputs to expect

## Autolog + Manual Logging

Autolog does not prevent custom logs — add whatever you need on top:

```python
mlflow.sklearn.autolog()

with mlflow.start_run():
    model.fit(X_train, y_train)                 # autolog fires here

    # Add custom metrics autolog doesn't know about
    mlflow.log_metric("test_mse", mean_squared_error(y_test, model.predict(X_test)))
    mlflow.log_artifact("residual_plot.png")
```

## GridSearchCV Child Runs

When you wrap a model in `GridSearchCV`, autolog creates one **child run per
CV trial** under the parent run:

```
Parent: gridsearch_ridge
  ├── Child: alpha=0.001  (cv_score, params)
  ├── Child: alpha=0.01
  ├── Child: alpha=0.1
  └── Child: alpha=1.0   ← best
```

`max_tuning_runs=N` caps child runs so the UI doesn't get cluttered with
dozens of trials from large grids.

## API Reference

```python
# Enable (call BEFORE training)
mlflow.sklearn.autolog()
mlflow.sklearn.autolog(
    log_models=True,              # save the trained model
    log_datasets=True,            # log a dataset profile
    log_input_examples=True,      # log sample inputs
    log_model_signatures=True,    # log input/output schema
    max_tuning_runs=5,            # cap GridSearchCV child runs
    log_post_training_metrics=True,
)

# Disable
mlflow.sklearn.autolog(disable=True)
mlflow.autolog(disable=True)      # all frameworks at once

# Other frameworks — same pattern
mlflow.xgboost.autolog()
mlflow.lightgbm.autolog()
mlflow.pytorch.autolog()
mlflow.tensorflow.autolog()
mlflow.statsmodels.autolog()
mlflow.autolog()                  # enable all supported frameworks
```

## Run It

```bash
python 03-autologging/autologging.py
mlflow ui
```

## What to Explore in the UI

1. Open `03_autologging` → **autolog_ridge** run
2. Params tab: see all the Ridge constructor args captured automatically
3. Artifacts tab: the model has an `input_example.json` and a `MLmodel` file
4. Click **gridsearch_ridge** → expand nested child runs for each CV trial

## Exercises

1. Add `mlflow.autolog()` (instead of `mlflow.sklearn.autolog()`) and train both Ridge and a RandomForest — confirm both are tracked
2. Disable `log_models=False` and check what's missing from the artifact store
3. Run a `GridSearchCV` with 20 parameter combinations — compare `max_tuning_runs=5` vs `max_tuning_runs=None`
4. Add a custom `log_artifact("plot.png")` call alongside autolog
5. Install `xgboost` and try `mlflow.xgboost.autolog()`

## Supported Frameworks

| Framework | Autolog function |
|-----------|-----------------|
| scikit-learn | `mlflow.sklearn.autolog()` |
| XGBoost | `mlflow.xgboost.autolog()` |
| LightGBM | `mlflow.lightgbm.autolog()` |
| PyTorch | `mlflow.pytorch.autolog()` |
| TensorFlow / Keras | `mlflow.tensorflow.autolog()` |
| statsmodels | `mlflow.statsmodels.autolog()` |
| All at once | `mlflow.autolog()` |

## Key Takeaways

- Call `autolog()` **before** `model.fit()` — that's the trigger point
- `log_input_examples=True` makes the model self-documenting (required for clean serving)
- `max_tuning_runs=N` prevents CV grids from flooding the UI with child runs
- Always disable with `autolog(disable=True)` when you're done to avoid side-effects
- You can always layer manual `log_metric` / `log_artifact` calls on top

## Next

**[Module 4: Model Registry →](../04-registry/README.md)**
