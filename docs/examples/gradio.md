## 🖼️ Cloud Inference with Gradio
You can further use Gradio to create a web interface for your model.

First, install the `dev` extra dependency.

```bash linenums="0"
pip install '.[dev]'
```

To use it, use an environment variable with your Focoos API key and run the app (you will select the model from the UI).
```bash linenums="0"
export FOCOOS_API_KEY_GRADIO=<YOUR-API-KEY>; python gradio/app.py
```
