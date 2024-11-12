# Focoos Foundational Models

| Model Name          | Task                  | Metrics | Domain                          |
| ------------------- | --------------------- | ------- | ------------------------------- |
| focoos_object365    | Detection             | -       | Common Objects, 365 classes     |
| focoos_rtdetr       | Detection             | -       | Common Objects, 80 classes      |
| focoos_cts_medium   | Semantic Segmentation | -       | Autonomous driving, 30 classes  |
| focoos_cts_large    | Semantic Segmentation | -       | Autonomous driving, 30 classes  |
| focoos_ade_nano     | Semantic Segmentation | -       | Common Scenes, 150 classes      |
| focoos_ade_small    | Semantic Segmentation | -       | Common Scenes, 150 classes      |
| focoos_ade_medium   | Semantic Segmentation | -       | Common Scenes, 150 classes      |
| focoos_ade_large    | Semantic Segmentation | -       | Common Scenes, 150 classes      |
| focoos_aeroscapes   | Semantic Segmentation | -       | Drone Aerial Scenes, 11 classes |
| focoos_isaid_nano   | Semantic Segmentation | -       | Satellite Imagery, 15 classes   |
| focoos_isaid_medium | Semantic Segmentation | -       | Satellite Imagery, 15 classes   |


# Focoos SDK

# Install

```bash
pip install .
```

- local inference (CPU, COREML): `pip install .[inference]`
- local inference with GPU (CUDA): `pip install .[inference-gpu]`

## 🤖 Cloud Inference
```python
from focoos import Focoos

focoos = Focoos(api_key=os.getenv("FOCOOS_API_KEY"))

model = focoos.get_remote_model("focoos_object365")
model.deploy()
detections = model.infer("./image.jpg", threshold=0.4)
```
## 🤖 Cloud Inference with Gradio

setup FOCOOS_API_KEY_GRADIO environment variable with your Focoos API key

```bash
pip install .[gradio]
```

```bash
python gradio/app.py
```

## Local Inference
```python
from focoos import Focoos

focoos = Focoos(api_key=os.getenv("FOCOOS_API_KEY"))

model = focoos.get_local_model("focoos_object365")

detections = model.infer("./image.jpg", threshold=0.4)
```
