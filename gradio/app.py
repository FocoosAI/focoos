import os

import gradio as gr
from focoos.model_manager import ModelManager
from focoos.model_registry import ModelRegistry
from focoos.utils.vision import annotate_image

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__)) + "/assets"

model_registry = ModelRegistry()

focoos_models = list(model_registry.list_models())

loaded_models = {}
image_examples = [
    ["fai-detr-l-coco", f"{ASSETS_DIR}/pexels-abby-chung.jpg"],
    ["fai-detr-l-obj365", f"{ASSETS_DIR}/motogp.jpg"],
    ["fai-detr-m-coco", f"{ASSETS_DIR}/ADE_val_00000821.jpg"],
    ["fai-mf-m-ade", f"{ASSETS_DIR}/ADE_val_00000461.jpg"],
    ["fai-mf-l-coco-ins", f"{ASSETS_DIR}/ADE_val_00000034.jpg"],
]


def run_inference(model_name, image, conf):
    if model_name not in loaded_models:
        model = ModelManager.get(model_name)
        loaded_models[model_name] = model
    else:
        model = loaded_models[model_name]
    detections = model(image, threshold=conf)
    annotated_image = annotate_image(image, detections, task=model.task, classes=model.classes)
    return annotated_image, detections.model_dump()


with gr.Blocks() as demo:
    gr.Markdown("## 🔥 Inference Focoos Pretrained Models")
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="numpy")
            model_name = gr.Dropdown(
                choices=list(focoos_models),
                label="Model",
                value=list(focoos_models)[0],
            )
            conf = gr.Slider(maximum=0.9, minimum=0, value=0.5, label="Confidence threshold")
            start_btn = gr.Button("Run Inference")
        with gr.Column():
            output_image = gr.Image(type="pil")
            output_detections = gr.JSON()
    examples = gr.Examples(
        fn=run_inference,
        inputs=[model_name, image, conf],
        outputs=[output_image],
        examples=image_examples,
    )
    start_btn.click(
        fn=run_inference,
        inputs=[model_name, image, conf],
        outputs=[output_image, output_detections],
    )


demo.launch()
