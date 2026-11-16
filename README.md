# MLflow Mastery

> A complete, hands-on curriculum for MLflow — from your first tracked run
> to production-ready serving, evaluation, and hyperparameter tuning.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.14%2B-0194E2?logo=mlflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Modules](https://img.shields.io/badge/Modules-10-orange)

---

## What Is This?

**10 self-contained modules**, each with a concept guide and a runnable Python
script. You go from zero (what is a run?) to production patterns (pyfunc
models, hyperparameter tuning, REST serving) in a structured sequence.

Every script runs top-to-bottom and logs real data to MLflow so you can
explore the UI as you learn. No toy "hello world" — each module uses patterns
you'd actually use in a real ML project.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/mlflow-mastery.git
cd mlflow-mastery

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the MLflow UI (leave this running in a separate terminal)
mlflow ui
# Open http://localhost:5000

# 4. Run Module 1
python 01-tracking/tracking.py
```

---

## The 10 Modules

| # | Module | What You'll Master |
|---|--------|--------------------|
| 1 | [**Tracking**](./01-tracking/README.md) | Runs, params, metrics, step curves, matplotlib artifacts, nested runs |
| 2 | [**Experiments**](./02-experiments/README.md) | Grouping runs, `search_runs()` with filters, `MlflowClient` |
| 3 | [**Autologging**](./03-autologging/README.md) | Zero-effort tracking for sklearn/XGBoost/etc, `GridSearchCV` child runs |
| 4 | [**Model Registry**](./04-registry/README.md) | Versioning, aliases vs stages, lifecycle, dynamic version lookup |
| 5 | [**Serving**](./05-serving/README.md) | REST API deployment, all input formats, batch inference |
| 6 | [**Projects**](./06-projects/README.md) | Reproducible packaging, typed parameters, `mlflow run` |
| 7 | [**Pipeline**](./07-pipeline/README.md) | Full end-to-end production workflow |
| 8 | [**Evaluation**](./08-evaluation/README.md) | `mlflow.evaluate()`, custom metrics, baseline comparison |
| 9 | [**Custom Models**](./09-custom-models/README.md) | `pyfunc` wrappers, preprocessing inside the model |
| 10 | [**Hyperparameter Tuning**](./10-hyperparameter/README.md) | Nested runs, grid/random search, winner promotion |

---

## What Makes This Different

Most MLflow tutorials stop at basic `log_param` / `log_metric`. This
curriculum goes further:

- **Step-based metrics** — log training curves, not just final values
- **Nested runs** — model hyperparameter sweeps properly
- **Aliases over stages** — the modern Registry API, not the deprecated one
- **Dynamic version lookup** — never hardcode version numbers
- **`mlflow.evaluate()`** — systematic model comparison with custom metrics
- **pyfunc models** — bundle preprocessing inside the model for safe serving
- **Baseline comparison** — measure how much your model actually adds

---

## Prerequisites

- Python 3.8+
- `pip install -r requirements.txt`
- No GPU required — all examples use scikit-learn on synthetic data

---

## How to Use This

```
1. Work through modules 1–10 in order
2. Read the README.md in each folder first (5 min)
3. Run the .py script and watch the output
4. Open mlflow ui and explore what was logged
5. Complete the exercises at the bottom of each README
```

Each module takes 20–40 minutes. The full curriculum is ~4–6 hours.

---

## Module Architecture

```
                     MLflow Tracking
              (runs, params, metrics, artifacts)
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
  MLflow Models      MLflow Registry    MLflow Projects
  (save / load)     (version / alias)  (reproducible)
        │                  │
  MLflow Serving      MLflow Evaluate
  (REST API)          (model analysis)
        │
  Custom pyfunc
  (your own logic)
```

---

## Repository Structure

```
mlflow-mastery/
├── requirements.txt
├── CHEATSHEET.md          ← complete API reference (bookmark this)
├── 01-tracking/
│   ├── README.md
│   └── tracking.py
├── 02-experiments/
│   ├── README.md
│   └── experiments.py
├── 03-autologging/
│   ├── README.md
│   └── autologging.py
├── 04-registry/
│   ├── README.md
│   └── registry.py
├── 05-serving/
│   ├── README.md
│   └── serving.py
├── 06-projects/
│   ├── README.md
│   ├── MLproject
│   ├── conda.yaml
│   └── train.py
├── 07-pipeline/
│   ├── README.md
│   └── pipeline.py
├── 08-evaluation/
│   ├── README.md
│   └── evaluation.py
├── 09-custom-models/
│   ├── README.md
│   └── custom_models.py
└── 10-hyperparameter/
    ├── README.md
    └── hyperparameter.py
```

---

## Quick Reference

```bash
mlflow ui                                              # launch tracking UI
mlflow models serve -m "models:/MyModel@champion" \
  --port 5001 --no-conda                              # serve a model
mlflow run 06-projects/ --no-conda \
  -P n_samples=500 -P alpha=0.1                       # run a project
```

See **[CHEATSHEET.md](./CHEATSHEET.md)** for the full API reference.

---

## Contributing

Contributions are welcome — new modules, bug fixes, better exercises.
See [CONTRIBUTING.md](./CONTRIBUTING.md) for how.

---

## License

[MIT](./LICENSE) — free to use, share, and adapt.

---

*Built to fill the gap between "MLflow hello world" and production usage.*
