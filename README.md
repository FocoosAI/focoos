# Focoos Foundational Models

| Model Name          | Task                  | Metrics | Domain                          |
|---------------------|-----------------------|---------|---------------------------------|
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
## 🤖 Cloud Inference
```python
from focoos import Focoos

focoos = Focoos(api_key=os.getenv("FOCOOS_API_KEY"))

model = focoos.get_model("focoos_object365")

detections = model.infer("./image.jpg", threshold=0.4)
```
