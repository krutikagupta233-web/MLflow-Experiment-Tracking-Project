# Module 9: Custom Models — MLflow pyfunc

## What You'll Learn

- What `mlflow.pyfunc` is and why it exists
- How to subclass `PythonModel` and implement `predict()`
- Bundling preprocessing logic inside the model
- `load_context()` — loading artifacts when the model is loaded
- Why custom pyfunc models are safer for production serving

## Why pyfunc Matters

MLflow has native "flavors" for sklearn, PyTorch, TensorFlow, XGBoost, etc.
But what if your model is:
- A scikit-learn model that needs a custom scaler that isn't saved with it?
- A rule-based scoring system?
- A third-party library that MLflow doesn't support natively?
- A pipeline that does feature engineering before calling a model?

`mlflow.pyfunc.PythonModel` solves all of these. It lets you define exactly
what happens when the model is loaded and when `.predict()` is called.

## The pyfunc Contract

```python
class MyModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Called ONCE when the model is loaded
        # Load artifacts from disk
        import pickle
        with open(context.artifacts["my_model_file"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input):
        # Called on every inference call
        # model_input is always a pandas DataFrame
        # Return a numpy array, pandas Series, or list
        return self.model.predict(model_input.values)
```

## Logging a pyfunc Model

```python
mlflow.pyfunc.log_model(
    artifact_path="my_model",         # folder name in the artifact store
    python_model=MyModel(),           # instance of your PythonModel subclass
    artifacts={
        "my_model_file": "/tmp/model.pkl",    # local file → artifact store
        "scaler":        "/tmp/scaler.pkl",
    },
    input_example=X_sample,           # helps with serving signature
)
```

The `artifacts` dict maps string keys to **local file paths**. MLflow copies
those files into the artifact store. Your `load_context()` reads them back
via `context.artifacts["key"]` (which gives the restored local path).

## Loading a pyfunc Model

```python
# Load by URI — works for any flavor, not just custom pyfunc
loaded = mlflow.pyfunc.load_model("runs:/<RUN_ID>/my_model")

# Always pass a DataFrame
import pandas as pd
preds = loaded.predict(pd.DataFrame(X_new, columns=feature_cols))
```

## Why Bundle Preprocessing

If your scaler is saved separately from your model, callers must:
1. Load the model
2. Load the scaler
3. Scale the input
4. Call the model

If anyone forgets step 3, predictions are silently wrong.

With a pyfunc wrapper, callers do one thing:
```python
model.predict(raw_df)   # scaling happens inside
```

This is the "self-contained model" principle: the model is responsible for
everything from raw input to raw output.

## API Reference

```python
import mlflow.pyfunc

# Define
class MyModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        ...  # load context.artifacts["key"]
    def predict(self, context, model_input):
        ...  # model_input is a DataFrame

# Log
mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=MyModel(),
    artifacts={"key": "/path/to/local/file"},
    input_example=df_sample,
)

# Load
loaded = mlflow.pyfunc.load_model("runs:/<RUN_ID>/model")
preds  = loaded.predict(pd.DataFrame(X_new, columns=cols))

# Load any flavor (sklearn, torch, etc.) generically
loaded = mlflow.pyfunc.load_model("models:/MyModel@champion")
```

## Run It

```bash
python 09-custom-models/custom_models.py
mlflow ui
```

## What to Explore in the UI

1. Open `09_custom_models` → any run
2. Artifacts tab → `scaled_model/` or `fe_model/` folder
3. Inside: `MLmodel` (flavor metadata), your `.pkl` files, `input_example.json`
4. The `MLmodel` file shows `flavor: python_function` — that's your pyfunc

## Exercises

1. Write a pyfunc model that applies `log1p` transformation to all features before predicting
2. Bundle a `LabelEncoder` with a classifier — encode predictions back to string labels in `predict()`
3. Write a `BlendedModel` pyfunc that averages predictions from two stored models
4. Log a pyfunc model to the Registry and set an alias for it
5. Serve it with `mlflow models serve` and call it via curl

## Key Takeaways

- `mlflow.pyfunc.PythonModel` wraps any Python logic as a loadable MLflow model
- `load_context()` runs once at load time — good for expensive deserialization
- `predict()` always receives a **pandas DataFrame** as `model_input`
- `artifacts` dict maps string keys to local file paths — MLflow copies them in
- Bundle preprocessing inside `predict()` to prevent silent ordering bugs
- `mlflow.pyfunc.load_model()` works for **all** MLflow flavors, not just custom ones

## Next

**[Module 10: Hyperparameter Tuning →](../10-hyperparameter/README.md)**
