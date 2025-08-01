import os
import uuid
from typing import Iterable

import cv2

import gradio as gr
from focoos import ASSETS_DIR, PREDICTIONS_DIR
from focoos.model_manager import ModelManager
from focoos.model_registry import ModelRegistry
from gradio.themes.base import Base, colors, fonts, sizes

os.makedirs(PREDICTIONS_DIR, exist_ok=True)


class FocoosTheme(Base):
    def __init__(
        self,
        primary_hue: colors.Color | str = colors.blue,
        secondary_hue: colors.Color | str = colors.emerald,
        neutral_hue: colors.Color | str = colors.zinc,
        text_size: sizes.Size | str = sizes.text_md,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.LocalFont("IBM Plex Sans"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.LocalFont("IBM Plex Mono"),
            "ui-monospace",
            "Consolas",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            spacing_size=spacing_size,
            radius_size=radius_size,
            font=font,
            font_mono=font_mono,
        )

        # Custom theme styling with blue-green gradient
        super().set(
            # Primary buttons with blue-green gradient
            button_primary_background_fill="linear-gradient(135deg, #0060e6 0%, #5fd49f 100%)",
            button_primary_background_fill_hover="linear-gradient(135deg, #5fd49f 0%, #0060e6 100%)",
            button_primary_background_fill_dark="linear-gradient(135deg, #0060e6 0%, #5fd49f 100%)",
            button_primary_text_color="white",
            button_primary_text_color_dark="white",
            # Slider with base colors (gradients added via CSS)
            slider_color="#0060e6",
            slider_color_dark="#0060e6",
            # Tab styling
            button_secondary_background_fill_hover="*primary_50",
            button_secondary_background_fill_hover_dark="*primary_900",
            # Progress bar base color (gradient added via CSS)
            loader_color="#0060e6",
            # Link colors
            link_text_color="#0060e6",
            link_text_color_hover="#5fd49f",
            link_text_color_dark="#0060e6",
            link_text_color_hover_dark="#5fd49f",
            # Checkbox and radio buttons
            checkbox_background_color_selected="#0060e6",
            checkbox_background_color_selected_dark="#0060e6",
            radio_circle="#0060e6",
            # Input focus
            input_border_color_focus="#0060e6",
            input_border_color_focus_dark="#0060e6",
            # Elegant borders and shadows
            block_border_width="1px",
            block_shadow="0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)",
        )


focoos_theme = FocoosTheme()

model_registry = ModelRegistry()
focoos_models = list(model_registry.list_models())
loaded_models = {}
image_examples = [
    [f"{ASSETS_DIR}/pexels-abby-chung.jpg", "fai-detr-l-coco"],
    [f"{ASSETS_DIR}/motogp.jpg", "fai-detr-l-obj365"],
    [f"{ASSETS_DIR}/ADE_val_00000821.jpg", "fai-detr-m-coco"],
    [f"{ASSETS_DIR}/ADE_val_00000461.jpg", "fai-mf-m-ade"],
    [f"{ASSETS_DIR}/ADE_val_00000034.jpg", "fai-mf-l-coco-ins"],
    [f"{ASSETS_DIR}/federer.jpg", "rtmo-s-coco"],
]

html = """
<a href="https://www.focoos.ai" target="_blank">
  <img src="https://public.focoos.ai/library/focoos_banner.png" alt="FocoosAI" style="max-width:100%;">
</a>
"""


def run_inference(image, model_name: str, conf: float, progress=gr.Progress()):
    assert model_name is not None, "model_name is required"
    assert model_name in model_registry.list_models(), "model_name is not valid"
    progress(0, desc="Loading Model...")
    if model_name not in loaded_models:
        model = ModelManager.get(model_name)
        loaded_models[model_name] = model
    else:
        model = loaded_models[model_name]
    progress(0.3, desc="Run Inference...")
    res = model.infer(image, threshold=conf, annotate=True)
    progress(0.9, desc="Done!")
    return res.image, res.model_dump()


def run_video_inference(
    video_path: str,
    model_name: str,
    threshold: float,
    progress=gr.Progress(),
):
    assert video_path is not None, "video_path is required"
    assert model_name is not None, "model_name is required"
    assert model_name in model_registry.list_models(), "model_name is not valid"

    progress(0, desc="Loading Model...")
    if model_name not in loaded_models:
        model = ModelManager.get(model_name)
        loaded_models[model_name] = model
    else:
        model = loaded_models[model_name]

    cap = cv2.VideoCapture(video_path)

    # This means we will output mp4 videos
    video_codec = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    desired_fps = fps

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    desired_width = int(width)
    desired_height = int(height)

    print(
        f"video: {video_path} fps: {fps}, total_frames: {total_frames}, desired_fps: {desired_fps}, width: {desired_width}, height: {desired_height}"
    )

    progress(0.1, desc="Initializing video...")

    # Use UUID to create a unique video file
    output_video_name = f"{PREDICTIONS_DIR}/gradio_output_{uuid.uuid4()}.mp4"

    # Output Video
    output_video = cv2.VideoWriter(output_video_name, video_codec, desired_fps, (desired_width, desired_height))  # type: ignore

    iterating, frame = cap.read()
    n_frames = 0
    last_latency = None

    progress(0.15, desc="Processing frames...")

    while iterating:
        if not cap.isOpened():
            print("Video ended")
            break

        if frame is None:
            iterating, frame = cap.read()
            continue

        frame = cv2.resize(frame, (desired_width, desired_height))
        res = model.infer(frame, threshold=threshold, annotate=True)
        last_latency = res.latency.inference if res.latency is not None else None

        # Write frame directly to video
        output_video.write(res.image)  # type: ignore

        n_frames += 1

        # Update progress
        progress_value = 0.15 + (0.8 * n_frames / total_frames)
        progress(progress_value, desc=f"Processing frame {n_frames}/{total_frames}")

        iterating, frame = cap.read()

    progress(0.95, desc="Finalizing video...")

    cap.release()
    output_video.release()

    progress(1.0, desc="Completed!")
    print(f"Video processed: {output_video_name}, total frames: {n_frames}")

    return (
        output_video_name,
        {
            "total_frames": n_frames,
            "latency(ms)": last_latency,
        },
    )


image_interface = gr.Interface(
    fn=run_inference,
    inputs=[
        gr.Image(type="numpy"),
        gr.Dropdown(
            choices=list(focoos_models),
            label="Model",
            value=list(focoos_models)[0],
        ),
        gr.Slider(maximum=0.9, minimum=0, value=0.5, label="Confidence threshold"),
    ],
    outputs=[gr.Image(type="pil"), gr.JSON()],
    examples=image_examples,
    flagging_mode="never",
)

video_interface = gr.Interface(
    fn=run_video_inference,
    inputs=[
        gr.Video(),
        gr.Dropdown(label="model", choices=list(focoos_models), value=list(focoos_models)[0]),
        gr.Slider(label="confidence threshold", minimum=0, maximum=1, value=0.5),
    ],
    flagging_mode="never",
    outputs=[gr.Video(streaming=True, autoplay=True, format="mp4"), gr.JSON()],
    description="Upload a video to run inference",
)


def launch_gradio(share: bool = False):
    demo = gr.Blocks(title="Focoos Pretrained Models", theme=focoos_theme)

    with demo:
        gr.HTML(html)

        with gr.Tabs():
            with gr.Tab("Image Inference"):
                image_interface.render()

            with gr.Tab("Video Inference"):
                video_interface.render()

    demo.launch(share=share)


if __name__ == "__main__":
    launch_gradio()
