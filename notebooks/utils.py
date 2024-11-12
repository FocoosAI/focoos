import json
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from focoos.utils.vision import image_preprocess


def load_coco_annotations(path):
    with open(os.path.join(path, "_annotations.coco.json")) as f:
        return json.load(f)


def get_random_image_indices(coco, num_images):
    num_images = min(num_images, len(coco["images"]))
    return np.random.choice(len(coco["images"]), num_images, replace=False)


def create_category_colors(alpha=1.0):
    category_colors = {}
    for i in range(1, 256):
        hue = (i * 137.5) % 360
        hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
        category_colors[i] = np.array([rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, alpha])
    return category_colors


def setup_plot_grid(num_images, title=None):
    grid_size = int(np.ceil(np.sqrt(num_images)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    if title:
        fig.suptitle(title, fontsize=16)
    return axes.flat if num_images > 1 else [axes]


def display_detection(path, num_images=9, annotate=True):
    coco = load_coco_annotations(path)
    img_indices = get_random_image_indices(coco, num_images)
    category_colors = create_category_colors(alpha=1.0)
    axes_flat = setup_plot_grid(num_images)

    for idx, ax in zip(img_indices, axes_flat):
        img_info = coco["images"][idx]
        img_path = os.path.join(path, img_info["file_name"])
        img = Image.open(img_path)
        img_array = np.array(img)

        if annotate:
            for ann in coco["annotations"]:
                if ann["image_id"] == img_info["id"]:
                    x, y, w, h = (int(v) for v in ann["bbox"])
                    category_id = ann["category_id"]
                    color = category_colors.get(category_id, (255, 255, 255))
                    cv2.rectangle(img_array, (x, y), (x + w, y + h), color, 2)

        ax.imshow(img_array)
        ax.axis("off")

    for ax in axes_flat[num_images:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def display_instseg(path, num_images=9, annotate=True):
    coco = load_coco_annotations(path)
    img_indices = get_random_image_indices(coco, num_images)
    category_colors = create_category_colors(alpha=0.5)
    axes_flat = setup_plot_grid(num_images)

    for idx, ax in zip(img_indices, axes_flat):
        img_info = coco["images"][idx]
        img_path = os.path.join(path, img_info["file_name"])
        img = Image.open(img_path)

        if annotate:
            masks = {}
            for ann in coco["annotations"]:
                if ann["image_id"] == img_info["id"]:
                    category_id = ann["category_id"]
                    if category_id not in masks:
                        masks[category_id] = np.zeros(
                            (img_info["height"], img_info["width"])
                        )
                    for seg in ann["segmentation"]:
                        poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(masks[category_id], [poly], 1)

            for category_id, mask in masks.items():
                color = category_colors.get(category_id, (0, 0, 0, 0.5))
                h, w = mask.shape[-2:]
                mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
                ax.imshow(mask)

        ax.imshow(img)
        ax.axis("off")

    for ax in axes_flat[num_images:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def start_gradio(model, paths, allowed_paths=["/Users/fcdl94/Develop/focoos/data"]):
    import gradio as gr

    def run_inference(image, conf=0.5):
        # Load and resize the image
        resized, _ = image_preprocess(image, resize=640)  # Using standard 640 size
        # Save to temporary file
        tmp_path = (
            f"/Users/fcdl94/Develop/focoos/data/{os.path.basename(image)}_resized.jpg"
        )
        # resized is in CHW format, need to convert to HWC and uint8 for saving
        img_to_save = resized[0].transpose(1, 2, 0).astype(np.uint8)
        cv2.imwrite(tmp_path, img_to_save)
        image = tmp_path

        detections, annotated_image = model.infer(image, conf, annotate=True)
        os.remove(tmp_path)
        return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), detections.model_dump()

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="filepath")
                start_btn = gr.Button("Run Inference")
            with gr.Column():
                output_image = gr.Image(type="pil")
                output_detections = gr.JSON()
        examples = gr.Examples(
            fn=run_inference,
            inputs=[image],
            outputs=[output_image],
            examples=[
                paths[i] for i in random.sample(range(len(paths)), min(5, len(paths)))
            ],
        )
        start_btn.click(
            fn=run_inference,
            inputs=[image],
            outputs=[output_image, output_detections],
        )
    return demo.launch(allowed_paths=allowed_paths)
