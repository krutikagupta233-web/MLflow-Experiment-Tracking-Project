# Module 5: MLflow Serving — REST API Deployment

## What You'll Learn

- How `mlflow models serve` works
- All three input payload formats for the `/invocations` endpoint
- Python REST client using `requests`
- Direct batch inference (no server) using `mlflow.pyfunc.load_model()`
- When to use REST vs direct loading

## How It Works

```
POST /invocations
{ "dataframe_split": {...} }
        │
mlflow models serve (Flask/gunicorn)
        │
Your logged model (loaded by MLflow)
        │
{ "predictions": [...] }
```

MLflow starts a server, loads your model, and wraps it in an HTTP endpoint.
The same model URI formats you use in Python work directly with the CLI.

## Starting the Server

```bash
# From a run
mlflow models serve -m "runs:/<RUN_ID>/model" --port 5001 --no-conda

# From the registry (alias — preferred)
mlflow models serve -m "models:/MyModel@champion" --port 5001 --no-conda

# From the registry (stage — deprecated)
mlflow models serve -m "models:/MyModel/Production" --port 5001 --no-conda
```

`--no-conda` skips creating a Conda environment and uses your current Python.
Use this for local development on Windows where Conda may not be configured.

## Input Formats

The `/invocations` endpoint accepts three JSON formats:

### Format 1: `inputs` — raw list of lists
```json
{"inputs": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}
```
Simple but order-dependent — columns must match the model's expected order.

### Format 2: `dataframe_split` — named columns + rows (recommended)
```json
{
  "dataframe_split": {
    "columns": ["feature_0", "feature_1", "feature_2"],
    "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
  }
}
```
Named columns prevent silent order bugs. Use this in production.

### Format 3: `dataframe_records` — list of dicts
```json
{
  "dataframe_records": [
    {"feature_0": 1.0, "feature_1": 2.0, "feature_2": 3.0},
    {"feature_0": 4.0, "feature_1": 5.0, "feature_2": 6.0}
  ]
}
```
Most readable but verbose for large batches.

## REST vs Direct Loading

| | REST API (`mlflow models serve`) | Direct (`load_model`) |
|--|----------------------------------|----------------------|
| Use case | Real-time services, microservices | Batch jobs, notebooks |
| Overhead | HTTP server startup (~5s) | None |
| Input type | JSON over HTTP | DataFrame directly |
| Scaling | Can put behind load balancer | Single process |
| Language | Any HTTP client | Python only |

For batch inference scripts and notebooks, **always use direct loading** —
it's faster, simpler, and avoids the HTTP overhead.

## API Reference

```bash
# CLI
mlflow models serve -m <uri> --port 5001 --no-conda
mlflow models build-docker -m <uri> -n my-model-image  # Docker

# curl
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["f0","f1","f2"], "data": [[1,2,3]]}}'
```

```python
# Direct loading (no server)
import mlflow, pandas as pd

model = mlflow.pyfunc.load_model("models:/MyModel@champion")
preds = model.predict(pd.DataFrame(X_new, columns=["f0","f1","f2"]))

# REST client (server must be running)
import requests, json
resp = requests.post(
    "http://127.0.0.1:5001/invocations",
    headers={"Content-Type": "application/json"},
    data=json.dumps({"dataframe_split": {"columns": cols, "data": X.tolist()}}),
)
print(resp.json())
```

## Run It

```bash
# Terminal 1
python 05-serving/serving.py
# Copy the "Model URI:" line from the output

# Terminal 2 (replace RUN_ID with the one from Terminal 1)
mlflow models serve -m "runs:/<RUN_ID>/model" --port 5001 --no-conda

# Terminal 3 (once the server says "Listening at http://127.0.0.1:5001")
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[5.0, 3.0, 7.0]]}'
```

## Exercises

1. Try all three input formats with curl
2. Write a Python `requests` client that sends a batch of 10 inputs
3. Serve a model from the registry by alias (run Module 4 first)
4. Use `mlflow.pyfunc.load_model()` for batch inference on 500 rows
5. Compare predictions from the REST endpoint vs direct loading — they should match

## Key Takeaways

- `mlflow models serve` is one CLI command — no code required
- Use `dataframe_split` format for production calls (named columns, no order bugs)
- `--no-conda` skips Conda environment creation (use on Windows / dev machines)
- Direct loading with `mlflow.pyfunc.load_model()` is better for batch jobs
- `pyfunc` loader works for **any** MLflow flavor — sklearn, pytorch, custom

## Next

**[Module 6: Projects →](../06-projects/README.md)**
