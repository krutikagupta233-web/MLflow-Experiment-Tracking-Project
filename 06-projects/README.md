# Module 6: MLflow Projects — Reproducible Packaging

## What You'll Learn

- How MLflow Projects package code for exact reproducibility
- The `MLproject` file — entry points and typed parameters
- Running a project with `mlflow run` and overriding parameters
- How parameters flow from the CLI into your training script

## Why Projects Matter

Without Projects, running the same code twice may give different results
because of environment drift, different argument defaults, or missing
dependencies. MLflow Projects solve this by making the run specification
part of the code:

```bash
# Anyone on any machine gets the same run
mlflow run 06-projects/ --no-conda -P n_samples=500 -P alpha=0.1
```

## Project Structure

```
06-projects/
├── MLproject      ← defines entry points, parameters, environment
├── conda.yaml     ← pins Python + package versions
└── train.py       ← the actual training script (accepts CLI args)
```

### MLproject File

```yaml
name: my-project
conda_env: conda.yaml
entry_points:
  main:
    parameters:
      alpha:     {type: float, default: 1.0}
      n_samples: {type: int,   default: 300}
    command: "python train.py --alpha {alpha} --n_samples {n_samples}"
```

**Parameter types:** `int`, `float`, `str`, `bool`

Each `{param_name}` placeholder in `command` is substituted at run time.

### conda.yaml

```yaml
name: mlflow-project-env
channels:
  - defaults
dependencies:
  - python=3.10
  - pip:
    - mlflow>=2.14.0
    - scikit-learn>=1.4.0
    - numpy>=1.26.0
    - pandas>=2.1.0
```

Pins exact versions so the environment is reproducible across machines.

### train.py

Uses `argparse` to accept parameters from the CLI — this is what MLproject's
`command` string invokes:

```python
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=1.0)
args = parser.parse_args()

with mlflow.start_run():
    mlflow.log_param("alpha", args.alpha)
    ...
```

## Running a Project

```bash
# Run from the project directory (uses defaults)
mlflow run 06-projects/ --no-conda

# Override parameters
mlflow run 06-projects/ --no-conda -P n_samples=1000 -P noise_level=0.05 -P alpha=0.01

# Run a specific entry point (if you have multiple)
mlflow run 06-projects/ --no-conda -e main

# Run from a GitHub URL
mlflow run https://github.com/mlflow/mlflow-example -P alpha=0.5
```

`--no-conda` uses your current Python environment instead of creating a
Conda environment. Use this for local development.

## API Reference

```bash
mlflow run .                         # run with defaults
mlflow run . -P key=value            # override one parameter
mlflow run . --no-conda              # skip Conda env creation
mlflow run . -e entry_point_name     # specify entry point
mlflow run . --experiment-name exp   # log to specific experiment
mlflow run https://github.com/...    # run from remote Git repo
```

## Run It

```bash
# Default run
mlflow run 06-projects/ --no-conda

# Low noise, more data
mlflow run 06-projects/ --no-conda -P n_samples=1000 -P noise_level=0.1

# High noise, small data — watch R² drop
mlflow run 06-projects/ --no-conda -P n_samples=100 -P noise_level=3.0

# Tune alpha
mlflow run 06-projects/ --no-conda -P alpha=0.01
mlflow run 06-projects/ --no-conda -P alpha=10.0
```

Each run is tracked in MLflow automatically.

## Exercises

1. Add a `model_type` parameter (`"Ridge"` or `"Lasso"`) and branch in `train.py`
2. Add a second entry point `evaluate` that loads the registered model and prints metrics
3. Run the project 5 times with different alphas and compare in the UI
4. Add `--experiment-name` to the run command to organize runs into an experiment
5. Check what happens when you run without `--no-conda` (requires Conda installed)

## Key Takeaways

- `MLproject` + `conda.yaml` + your script = a fully reproducible run unit
- Parameters are typed and have defaults — override with `-P key=value`
- `--no-conda` skips Conda (use in dev; avoid in CI/CD where reproducibility matters)
- Every `mlflow run` automatically logs to the tracking server
- Projects can be run from local paths or remote Git URLs

## Next

**[Module 7: End-to-End Pipeline →](../07-pipeline/README.md)**
