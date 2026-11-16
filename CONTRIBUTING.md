# Contributing to mlflow-mastery

Contributions are welcome — bug fixes, new modules, clearer explanations,
or better exercises. Here's how.

## Ways to Contribute

- **Fix a bug** — script error, broken example, outdated API usage
- **Improve an explanation** — clearer README, better inline comments
- **Add exercises** — more practice problems to an existing module
- **Add a module** — new topics (see suggestions below)
- **Update for new MLflow versions** — API changes, new features

## Module Ideas (not yet covered)

- Remote tracking server setup (Postgres + S3/MinIO)
- MLflow with PyTorch training loops
- MLflow with TensorFlow/Keras
- Bayesian optimisation (Optuna + MLflow)
- Drift monitoring (Evidently + MLflow)
- Deploying to AWS SageMaker
- MLflow on Databricks
- MLflow Recipes (structured pipelines)

## Process

1. Fork the repo
2. Create a branch: `git checkout -b feature/module-11-remote-server`
3. Make your changes
4. Test your script: `python <module>/script.py`
5. Open a pull request with a clear description

## Script Standards

Each module should have:
- A `.py` script that runs top-to-bottom without errors
- A `README.md` explaining concepts, API, exercises, and key takeaways
- Clear inline comments for non-obvious lines
- A `print("=" * 60)` summary block at the end with key takeaways

## Reporting Issues

Open a GitHub issue with:
- The module number and script name
- The error message or description
- Your Python and MLflow versions (`python --version`, `mlflow --version`)
