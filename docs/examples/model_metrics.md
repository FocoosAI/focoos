

## 📈 Focoos Models metrics
You can see the metrics of the Focoos Models (or your own models) by calling the [`metrics` method](/focoos/api/remote_model/#focoos.remote_model.RemoteModel.metrics) on the model.

```python
model = focoos.get_remote_model("fai-rtdetr-m-obj365")
metrics = model.metrics()
print(f"Best validation metrics:")
for k, v in metrics.best_valid_metric.items():
    print(f"  {k}: {v}")
```
This code snippet will print the best validation metrics of the model, both considering average and per-class metrics.
