import os

import cv2
from dotenv import load_dotenv

import gradio as gr
from focoos import Focoos, FocoosEnvHostUrl

load_dotenv()
ASSETS_DIR = os.path.dirname(os.path.abspath(__file__)) + "/assets"

focoos_models = []
focoos = Focoos(api_key=os.getenv("FOCOOS_API_KEY"), host_url=FocoosEnvHostUrl.DEV)
focoos_models = [
    model["ref"]
    for model in focoos.list_focoos_models()
    if model["status"] == "DEPLOYED"
]
loaded_models = {}
image_examples = [
    ["focoos_rtdetr", f"{ASSETS_DIR}/pexels-abby-chung.jpg"],
    ["focoos_object365", f"{ASSETS_DIR}/motogp.jpg"],
    ["focoos_ade_medium", f"{ASSETS_DIR}/ADE_val_00000034.jpg"],
    ["focoos_cts_medium", f"{ASSETS_DIR}/frankfurt_000001_059789_leftImg8bit.jpg"],
    ["focoos_isaid_nano", f"{ASSETS_DIR}/P0053_0_896_512_1408.jpeg"],
    ["focoos_isaid_medium", f"{ASSETS_DIR}/P0161_hires.jpeg"],
    ["focoos_aeroscapes", f"{ASSETS_DIR}/aeroscapes.jpg"],
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
            conf = gr.Slider(
                maximum=0.9, minimum=0, value=0.5, label="Confidencte threshold"
            )
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
