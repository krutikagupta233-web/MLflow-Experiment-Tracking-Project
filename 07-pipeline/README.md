# Module 7: End-to-End MLflow Pipeline

## What You'll Learn

- How all MLflow components fit together in a production workflow
- The version number trap and how to avoid it
- Aliases for promotion (no deprecated stage transitions)
- Feature importance logging as metrics
- A complete template you can adapt for real projects

## The Pipeline

```
         Data
          │
   ┌──────┴──────┐
   Train 6 models  (each logged as a run + registered as a version)
   └──────┬──────┘
          │
   Find best by MSE
          │
   set_registered_model_alias(@champion)
          │
   load_model("models:/PipelineModel@champion")
          │
   Predictions
```

Every step is tracked. Every model version is in the Registry.
The best one gets the `@champion` alias. Your serving code never changes.

## The Version Number Trap

Every time you run `pipeline.py`, new versions are created. If you hardcode
`version=3` to promote the best model, it breaks on the second run.

The fix — always look up by `run_id`:

```python
# After training, find your run's version in the Registry
version_map = {mv.run_id: mv.version
               for mv in client.search_model_versions(f"name='{REGISTERED_NAME}'")}

champion_version = version_map[best_run_id]   # always correct
client.set_registered_model_alias(REGISTERED_NAME, "champion", champion_version)
```

## Production Checklist

```
[ ] Experiment created with descriptive tags
[ ] All params logged (model type, hyperparams, data info)
[ ] All metrics logged (MSE, RMSE, R², MAE)
[ ] Models registered via registered_model_name
[ ] Best model promoted via alias (not hardcoded version)
[ ] Model description and tags set
[ ] Champion version tested after loading
```

## Extending the Pipeline

This pipeline is a starting point. Real production adds:

| Need | Tool |
|------|------|
| Data validation | Great Expectations, Pandera |
| Feature store | Feast, Tecton |
| Model evaluation | `mlflow.evaluate()` (Module 8) |
| Drift monitoring | Evidently, WhyLabs |
| CI/CD | GitHub Actions + this script |
| Containerization | `mlflow models build-docker` |

## Run It

```bash
python 07-pipeline/pipeline.py
mlflow ui
```

Then:
```bash
mlflow models serve -m "models:/PipelineModel@champion" --port 5001 --no-conda
```

## Exercises

1. Add a `KNeighborsRegressor` to `model_configs`
2. Log training time as a metric (`time.time()` before and after `pipe.fit()`)
3. Add a `git_commit` tag using `subprocess.check_output(["git", "rev-parse", "HEAD"])`
4. After running, change the champion to the second-best model — observe what changes
5. Add `mlflow.evaluate()` after the champion is loaded (preview of Module 8)

## Key Takeaways

- `registered_model_name` in `log_model()` auto-creates versions
- `version_map = {mv.run_id: mv.version ...}` — safe version lookup
- `set_registered_model_alias()` is the modern alternative to `transition_model_version_stage()`
- Feature importances can be logged as individual metrics per feature
- Every step is auditable — the full training history lives in MLflow

## Next

**[Module 8: Evaluation →](../08-evaluation/README.md)**
