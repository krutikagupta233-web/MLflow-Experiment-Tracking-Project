# MLflow Complete Cheatsheet

## 1. Tracking

```python
import mlflow

# Context manager (always prefer this)
with mlflow.start_run(run_name="my_run") as run:
    print(run.info.run_id)          # unique run identifier

    # Parameters — inputs/config (immutable once logged)
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_params({"epochs": 100, "batch_size": 32, "optimizer": "adam"})

    # Metrics — numeric results (can be updated, support step)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("loss", 0.05, step=10)   # step = epoch/iteration number
    mlflow.log_metrics({"f1": 0.93, "precision": 0.94, "recall": 0.92})

    # Log a metric over training steps (creates a time-series chart in UI)
    for epoch in range(100):
        mlflow.log_metric("train_loss", loss_value, step=epoch)

    # Tags — free-form metadata (searchable)
    mlflow.set_tag("team", "ml-eng")
    mlflow.set_tags({"env": "prod", "dataset": "v2", "owner": "alice"})

    # Artifacts — files
    mlflow.log_artifact("confusion_matrix.png")       # single file
    mlflow.log_artifact("confusion_matrix.png", "plots")  # into subfolder
    mlflow.log_artifact("outputs/")                   # entire directory
    mlflow.log_text("some notes", "notes.txt")        # string → file
    mlflow.log_dict({"key": "value"}, "config.json")  # dict → JSON

    # Log matplotlib figure directly
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    mlflow.log_figure(fig, "my_plot.png")
    plt.close(fig)

    # Models
    mlflow.sklearn.log_model(model, name="model", input_example=X[:3])
    mlflow.pyfunc.log_model("model", python_model=MyPyfuncModel())

# Nested runs (parent → children)
with mlflow.start_run(run_name="sweep") as parent:
    for alpha in [0.1, 1.0, 10.0]:
        with mlflow.start_run(run_name=f"alpha_{alpha}", nested=True):
            mlflow.log_param("alpha", alpha)
```

---

## 2. Experiments

```python
# Create / select
mlflow.create_experiment("my_exp", tags={"team": "ml"})
mlflow.set_experiment("my_exp")

# Get info
exp = mlflow.get_experiment_by_name("my_exp")
print(exp.experiment_id, exp.name, exp.artifact_location)

# Search runs — returns a pandas DataFrame
df = mlflow.search_runs(experiment_names=["my_exp"])

# Filter and sort
df = mlflow.search_runs(
    experiment_names=["my_exp"],
    filter_string='metrics.mse < 0.1 AND tags.env = "prod"',
    order_by=["metrics.mse ASC"],
    max_results=50,
)

# Search by run status
df = mlflow.search_runs(filter_string='status = "FINISHED"')

# Find the best run
best_run_id = df.loc[df["metrics.mse"].idxmin(), "run_id"]

# Access run metadata
from mlflow.tracking import MlflowClient
client = MlflowClient()
run = client.get_run(run_id)
print(run.data.params, run.data.metrics, run.data.tags)

# List all experiments
for exp in client.search_experiments():
    print(exp.experiment_id, exp.name)
```

---

## 3. Autologging

```python
# Enable (call BEFORE model.fit())
mlflow.sklearn.autolog()                         # sklearn
mlflow.xgboost.autolog()                         # XGBoost
mlflow.lightgbm.autolog()                        # LightGBM
mlflow.pytorch.autolog()                         # PyTorch
mlflow.tensorflow.autolog()                      # TensorFlow/Keras
mlflow.statsmodels.autolog()                     # statsmodels
mlflow.autolog()                                 # all supported frameworks

# Fine-grained control
mlflow.sklearn.autolog(
    log_models=True,           # save the trained model
    log_datasets=True,         # log dataset profile
    log_input_examples=True,   # log sample inputs
    log_model_signatures=True, # log input/output schema
    max_tuning_runs=10,        # for CV / GridSearchCV
)

# Disable
mlflow.sklearn.autolog(disable=True)
mlflow.autolog(disable=True)
```

---

## 4. Model Registry

```python
from mlflow.tracking import MlflowClient
client = MlflowClient()

# Register a model when logging
mlflow.sklearn.log_model(
    model, "model",
    registered_model_name="MyModel",   # this creates the registry entry
    input_example=X[:3],
)

# Register an existing run's model after the fact
mlflow.register_model(f"runs:/{run_id}/model", "MyModel")

# ── Aliases (modern, preferred over stages) ──────────────────────────────
client.set_registered_model_alias("MyModel", "champion", version=3)
client.set_registered_model_alias("MyModel", "challenger", version=4)
client.delete_registered_model_alias("MyModel", "challenger")

# ── Stage transitions (deprecated but widely used in legacy code) ─────────
client.transition_model_version_stage("MyModel", 2, "Staging")
client.transition_model_version_stage("MyModel", 2, "Production")
client.transition_model_version_stage("MyModel", 1, "Archived")

# ── Load models ──────────────────────────────────────────────────────────
mlflow.sklearn.load_model("models:/MyModel@champion")   # by alias (preferred)
mlflow.sklearn.load_model("models:/MyModel/3")          # by version number
mlflow.sklearn.load_model("models:/MyModel/Production") # by stage (deprecated)
mlflow.pyfunc.load_model("models:/MyModel@champion")    # generic (any flavor)

# ── Search & inspect ──────────────────────────────────────────────────────
for mv in client.search_model_versions("name='MyModel'"):
    print(mv.version, mv.current_stage, mv.aliases, mv.run_id)

# Find version for a specific run_id
versions = {mv.run_id: mv.version
            for mv in client.search_model_versions("name='MyModel'")}

# Add metadata
client.update_model_version("MyModel", 2, description="Best Ridge model")
client.set_model_version_tag("MyModel", 2, "approved_by", "alice")
client.update_registered_model("MyModel", description="Regression model v2")

# Delete
client.delete_model_version("MyModel", 1)
client.delete_registered_model("MyModel")   # deletes ALL versions
```

---

## 5. Model URI Formats

```
runs:/<RUN_ID>/model           # from a specific run
models:/MyModel@champion       # by alias (preferred)
models:/MyModel/3              # by version number
models:/MyModel/Production     # by stage (deprecated)
```

---

## 6. Serving

```bash
# Start REST server (--no-conda skips Conda env creation)
mlflow models serve -m "models:/MyModel@champion" --port 5001 --no-conda
mlflow models serve -m "runs:/<RUN_ID>/model"     --port 5001 --no-conda

# Test with curl
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[1.0, 2.0, 3.0]]}'

# Pandas split format
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["f0","f1"], "data": [[1.0, 2.0]]}}'
```

```python
# Python client (no server needed — load directly)
import mlflow
model = mlflow.sklearn.load_model("models:/MyModel@champion")
preds = model.predict(X_new)

# REST client
import requests, json
response = requests.post(
    "http://127.0.0.1:5001/invocations",
    headers={"Content-Type": "application/json"},
    data=json.dumps({"inputs": X_new.tolist()}),
)
print(response.json())
```

---

## 7. Projects

```yaml
# MLproject
name: my-project
conda_env: conda.yaml
entry_points:
  main:
    parameters:
      lr:      {type: float, default: 0.01}
      epochs:  {type: int,   default: 100}
    command: "python train.py --lr {lr} --epochs {epochs}"
  evaluate:
    parameters:
      run_id: {type: str}
    command: "python evaluate.py --run_id {run_id}"
```

```bash
mlflow run .                          # run with defaults
mlflow run . -P lr=0.001 -P epochs=50 # override params
mlflow run https://github.com/user/repo -P lr=0.001  # from GitHub
```

---

## 8. Evaluate

```python
import mlflow

result = mlflow.evaluate(
    model="runs:/<RUN_ID>/model",  # or a Python callable
    data=eval_df,                   # DataFrame with features + target column
    targets="target",              # target column name
    model_type="regressor",        # or "classifier"
    evaluators="default",
)

# Built-in regression metrics
print(result.metrics["root_mean_squared_error"])
print(result.metrics["mean_absolute_error"])
print(result.metrics["r2_score"])
print(result.metrics["mean_absolute_percentage_error"])

# Built-in classifier metrics
# accuracy_score, f1_score, precision_score, recall_score, roc_auc, log_loss

# Custom metric
from mlflow.metrics import make_metric, MetricValue

def my_metric(predictions, targets, metrics, sample_weights=None):
    import numpy as np
    val = float(np.mean(np.abs(targets.values - predictions.values)))
    return MetricValue(aggregate_results={"mean": val})

custom = make_metric(eval_fn=my_metric, name="my_metric", greater_is_better=False)

result = mlflow.evaluate(..., extra_metrics=[custom])
```

---

## 9. Custom pyfunc Models

```python
class MyModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import pickle
        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)
        with open(context.artifacts["scaler"], "rb") as f:
            self.scaler = pickle.load(f)

    def predict(self, context, model_input):
        X = self.scaler.transform(model_input)
        return self.model.predict(X)

# Log
mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=MyModel(),
    artifacts={"model": "/tmp/model.pkl", "scaler": "/tmp/scaler.pkl"},
    input_example=X_sample,
)

# Load
loaded = mlflow.pyfunc.load_model("runs:/<RUN_ID>/model")
preds = loaded.predict(X_new_df)   # always pass a DataFrame
```

---

## 10. CLI Commands

```bash
mlflow ui                                  # launch UI (port 5000)
mlflow ui --port 5001                      # custom port
mlflow ui --host 0.0.0.0                   # expose to network

mlflow run .                               # run MLproject
mlflow models serve -m <uri> --port 5001 --no-conda
mlflow models build-docker -m <uri> -n myimage  # build Docker image

# Programmatic search
python -c "import mlflow; print(mlflow.search_runs())"
```

---

## Key Classes

| Symbol | Purpose |
|--------|---------|
| `mlflow.start_run()` | Context manager — creates a tracked run |
| `MlflowClient` | Low-level client for tracking server + registry |
| `mlflow.pyfunc` | Generic model interface — load any flavor |
| `mlflow.evaluate()` | Systematic model evaluation with built-in + custom metrics |
| `mlflow.metrics.make_metric()` | Define a custom evaluation metric |
