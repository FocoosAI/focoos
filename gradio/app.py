import os

import cv2

import gradio as gr
from focoos import FocoosHUB

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__)) + "/assets"

focoos_models = []
focoos = FocoosHUB(api_key=os.getenv("FOCOOS_API_KEY"))
focoos_models = [model.ref for model in focoos.list_pretrained_models()]
loaded_models = {}
image_examples = [
    ["fai-rtdetr-l-coco", f"{ASSETS_DIR}/pexels-abby-chung.jpg"],
    ["fai-rtdetr-m-obj365", f"{ASSETS_DIR}/motogp.jpg"],
    ["fai-rtdetr-s-coco", f"{ASSETS_DIR}/ADE_val_00000821.jpg"],
    ["fai-m2f-m-ade", f"{ASSETS_DIR}/ADE_val_00000461.jpg"],
    ["fai-m2f-l-coco-ins", f"{ASSETS_DIR}/ADE_val_00000034.jpg"],
]


def run_inference(model_name, image, conf):
    if not model_name or not image or not conf:
        raise gr.Error("Model name and image are required")
    if model_name not in loaded_models:
        model = focoos.get_remote_model(model_name)
        loaded_models[model_name] = model
    else:
        model = loaded_models[model_name]
    detections, annotated_image = model.infer(image, conf, annotate=True)
    return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), detections.model_dump()


with gr.Blocks() as demo:
    gr.Markdown("## 🔥  Cloud Inference Focoos Foundational Models")
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="filepath")
            model_name = gr.Dropdown(
                choices=list(focoos_models),
                label="Model",
                value=list(focoos_models)[0],
            )
            conf = gr.Slider(maximum=0.9, minimum=0, value=0.5, label="Confidencte threshold")
            start_btn = gr.Button("Run Inference")
        with gr.Column():
            output_image = gr.Image(type="pil")
            output_detections = gr.JSON()
    examples = gr.Examples(
        fn=run_inference,
        inputs=[model_name, image],
        outputs=[output_image],
        examples=image_examples,
    )
    start_btn.click(
        fn=run_inference,
        inputs=[model_name, image, conf],
        outputs=[output_image, output_detections],
    )


demo.launch()
