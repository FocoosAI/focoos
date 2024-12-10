# Inferece

This section covers how to perform inference using the `focoos` library. You can deploy models to the cloud for predictions, integrate with Gradio for interactive demos, or run inference locally.

---

## 🤖 Cloud Inference

```python
from focoos import Focoos

focoos = Focoos(api_key=os.getenv("FOCOOS_API_KEY"))

model = focoos.get_remote_model("focoos_object365")
detections = model.infer("./image.jpg", threshold=0.4)
```

## 🤖 Cloud Inference with Gradio

setup `FOCOOS_API_KEY_GRADIO` environment variable with your Focoos API key

```bash linenums="0"
pip install '.[gradio]'
```

```bash linenums="0"
python gradio/app.py
```

## 🤖 Local Inference

```python
from focoos import Focoos

focoos = Focoos(api_key=os.getenv("FOCOOS_API_KEY"))

model = focoos.get_local_model("focoos_object365")

detections = model.infer("./image.jpg", threshold=0.4)
```
