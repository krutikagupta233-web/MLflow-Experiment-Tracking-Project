"""
================================================================================
MODULE 5: MLflow Serving — Deploy a Model as a REST API
================================================================================
MLflow can turn any logged model into a REST API endpoint with one CLI command.
The same model URI system from the Registry works directly here.

This module covers:
  A. Train and log a model, get its URI
  B. Serve it as a REST API (instructions + curl commands)
  C. Call it from Python with requests
  D. Batch inference without a server (load directly)
  E. All input formats: inputs, dataframe_split, dataframe_records
================================================================================
"""

import json

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ── Data ──────────────────────────────────────────────────────────────────────
np.random.seed(42)
X = np.random.rand(200, 3) * 10
y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 3.0 * X[:, 2] + 5 + np.random.randn(200) * 0.3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
feature_cols = ["feature_0", "feature_1", "feature_2"]

# ══════════════════════════════════════════════════════════════════════════════
# A. Train and log the model
# ══════════════════════════════════════════════════════════════════════════════
model = LinearRegression()
model.fit(X_train, y_train)
r2  = r2_score(y_test, model.predict(X_test))
mse = mean_squared_error(y_test, model.predict(X_test))
print(f"Trained model — R²={r2:.4f}  MSE={mse:.4f}")

mlflow.set_experiment("05_serving")

with mlflow.start_run(run_name="model_for_serving") as run:
    mlflow.log_metrics({"r2": r2, "mse": mse})

    # input_example + signature are critical for serving — they document
    # what format the REST endpoint expects
    input_df = pd.DataFrame(X_train[:5], columns=feature_cols)
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=input_df,
    )
    model_uri = f"runs:/{run.info.run_id}/model"

print(f"\nModel URI:  {model_uri}")


# ══════════════════════════════════════════════════════════════════════════════
# B. How to start the REST server
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("  START THE SERVER (open a NEW terminal):")
print("=" * 64)
print(f'  mlflow models serve -m "{model_uri}" --port 5001 --no-conda')
print()
print("  Wait for 'Listening at: http://127.0.0.1:5001'")
print()


# ══════════════════════════════════════════════════════════════════════════════
# C. curl commands — all input formats
# ══════════════════════════════════════════════════════════════════════════════
sample_data = X_test[:3].tolist()

inputs_payload = json.dumps({"inputs": sample_data}, indent=2)
split_payload  = json.dumps({
    "dataframe_split": {
        "columns": feature_cols,
        "data": sample_data,
    }
}, indent=2)
records_payload = json.dumps({
    "dataframe_records": [
        {col: val for col, val in zip(feature_cols, row)}
        for row in sample_data
    ]
}, indent=2)

print("=" * 64)
print("  TEST WITH CURL (three equivalent input formats):")
print("=" * 64)
print()
print("  # Format 1: inputs (raw array of arrays)")
print(f"""  curl -X POST http://127.0.0.1:5001/invocations \\
    -H "Content-Type: application/json" \\
    -d '{json.dumps({"inputs": sample_data})}'""")
print()
print("  # Format 2: dataframe_split (named columns + rows)")
print(f"""  curl -X POST http://127.0.0.1:5001/invocations \\
    -H "Content-Type: application/json" \\
    -d '{json.dumps({"dataframe_split": {"columns": feature_cols, "data": sample_data}})}'""")
print()
print("  # Format 3: dataframe_records (list of dicts)")
sample_records = [{col: val for col, val in zip(feature_cols, row)} for row in sample_data]
print(f"""  curl -X POST http://127.0.0.1:5001/invocations \\
    -H "Content-Type: application/json" \\
    -d '{json.dumps({"dataframe_records": sample_records})}'""")


# ══════════════════════════════════════════════════════════════════════════════
# D. Python REST client (server must be running)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("  PYTHON REST CLIENT (paste this into a separate script):")
print("=" * 64)

python_client = f'''
import requests, json, numpy as np

URL = "http://127.0.0.1:5001/invocations"

# Using dataframe_split format (most readable)
payload = {{
    "dataframe_split": {{
        "columns": {feature_cols},
        "data": {X_test[:5].tolist()},
    }}
}}

response = requests.post(URL, headers={{"Content-Type": "application/json"}}, data=json.dumps(payload))
print("Status:", response.status_code)
print("Predictions:", response.json())
'''
print(python_client)


# ══════════════════════════════════════════════════════════════════════════════
# E. Direct Python loading — no server needed (batch inference)
#    This is the recommended approach for batch jobs (ETL, notebooks, etc.)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("  LOCAL BATCH INFERENCE (no server needed):")
print("=" * 64)

# Load via pyfunc — works for any MLflow flavor
loaded = mlflow.pyfunc.load_model(model_uri)

# pyfunc always expects a DataFrame
input_df = pd.DataFrame(X_test, columns=feature_cols)
preds = loaded.predict(input_df)

print(f"\n  Loaded model from: {model_uri}")
print(f"  Batch size: {len(input_df)} samples")
print(f"  Prediction sample:  {preds[:5].tolist()}")
print(f"  Actual sample:      {y_test[:5].tolist()}")
print(f"  MSE on test set:    {mean_squared_error(y_test, preds):.4f}")


print()
print("=" * 64)
print("KEY TAKEAWAYS:")
print("  mlflow models serve -m <uri> --port 5001 --no-conda")
print("  POST to /invocations — three supported JSON formats")
print("  inputs:            raw list-of-lists (column-order matters)")
print("  dataframe_split:   named columns + data rows (safest)")
print("  dataframe_records: list of dicts (most readable)")
print("  mlflow.pyfunc.load_model() — direct load, no server, works for all flavors")
print("  Use direct loading for batch jobs; REST API for real-time services")
