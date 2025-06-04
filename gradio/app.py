import os
import uuid

import cv2

import gradio as gr
from focoos.model_manager import ModelManager
from focoos.model_registry import ModelRegistry
from focoos.utils.vision import annotate_frame, annotate_image

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__)) + "/assets"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SUBSAMPLE = 2


model_registry = ModelRegistry()
focoos_models = list(model_registry.list_models())
loaded_models = {}
image_examples = [
    [f"{ASSETS_DIR}/pexels-abby-chung.jpg", "fai-detr-l-coco"],
    [f"{ASSETS_DIR}/motogp.jpg", "fai-detr-l-obj365"],
    [f"{ASSETS_DIR}/ADE_val_00000821.jpg", "fai-detr-m-coco"],
    [f"{ASSETS_DIR}/ADE_val_00000461.jpg", "fai-mf-m-ade"],
    [f"{ASSETS_DIR}/ADE_val_00000034.jpg", "fai-mf-l-coco-ins"],
]


def run_inference(image, model_name: str, conf: float, progress=gr.Progress()):
    assert model_name is not None, "model_name is required"
    assert model_name in model_registry.list_models(), "model_name is not valid"
    if model_name not in loaded_models:
        model = ModelManager.get(model_name)
        loaded_models[model_name] = model
    else:
        model = loaded_models[model_name]
    detections = model(image, threshold=conf)
    annotated_image = annotate_image(image, detections, task=model.task, classes=model.classes)
    return annotated_image, detections.model_dump()


def run_video_inference(
    video_path: str,
    model_name: str,
    threshold: float,
    progress=gr.Progress(),
):
    assert video_path is not None, "video_path is required"
    assert model_name is not None, "model_name is required"
    assert model_name in model_registry.list_models(), "model_name is not valid"

    progress(0, desc="Load Model...")
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
    output_video_name = f"{OUTPUT_DIR}/output_{uuid.uuid4()}.mp4"

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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = model(frame, threshold=threshold)
        last_latency = res.latency.get("inference") if res.latency is not None else None

        annotated_frame = annotate_frame(frame, res, task=model.task, classes=model.classes)

        # Write frame directly to video
        output_video.write(annotated_frame[:, :, ::-1])

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


demo = gr.TabbedInterface(
    title="Focoos Pretrained Models",
    interface_list=[image_interface, video_interface],
    tab_names=["Image Inference", "Video Inference"],
)
demo.launch()
