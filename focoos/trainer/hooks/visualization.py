# Copyright (c) FocoosAI
import math
import os
import random
from contextlib import ExitStack, contextmanager
from typing import Optional

import cv2
import numpy as np
import torch

from focoos.data.datasets.map_dataset import MapDataset
from focoos.models.focoos_model import BaseModelNN
from focoos.processor.base_processor import Processor
from focoos.trainer.events import get_event_storage
from focoos.utils.logger import get_logger
from focoos.utils.visualizer import ColorMode, Visualizer

from .base import HookBase

logger = get_logger("VisualizationHook")


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


class VisualizationHook(HookBase):
    def __init__(
        self,
        model: BaseModelNN,
        processor: Processor,
        dataset: MapDataset,
        period,
        index_list=[],
        n_sample=4,
        random_samples=False,
        output_dir: Optional[str] = None,
    ):
        self.model = model
        self.processor = processor
        # self.postprocessing = postprocessing
        self._period = period
        if index_list is None or len(index_list) <= 0:
            if random_samples:
                index_list = random.sample(range(len(dataset)), k=min(len(dataset), n_sample))
            else:
                index_list = [i for i in range(min(len(dataset), n_sample))]
        self.n_sample = len(index_list)

        self.samples = [dataset[i] for i in index_list]
        self.metadata = dataset.dataset.metadata
        self.cpu_device = torch.device("cpu")
        self.output_dir = output_dir

    def _create_mosaic(self, images):
        """
        Create a mosaic of images with max resolution 3000x3000.

        Args:
            images: list of numpy arrays with shape (H, W, 3)

        Returns:
            mosaic: numpy array with shape (H, W, 3)
        """
        if not images:
            logger.warning("No images to create mosaic")
            return None

        # Calculate optimal grid dimensions
        n_images = len(images)
        grid_size = math.ceil(math.sqrt(n_images))
        grid_rows = math.ceil(n_images / grid_size)
        grid_cols = grid_size

        # Check image sizes
        heights = [img.shape[0] for img in images]
        widths = [img.shape[1] for img in images]

        # Calculate target size for each image to fit in 3000x3000
        max_size = 3000
        target_height = min(int(max_size / grid_rows), max(heights))
        target_width = min(int(max_size / grid_cols), max(widths))

        # Resize images to target size
        resized_images = []
        for img in images:
            # Preserve aspect ratio
            h, w = img.shape[:2]
            ratio = min(target_height / h, target_width / w)
            new_h, new_w = int(h * ratio), int(w * ratio)
            resized = cv2.resize(img, (new_w, new_h))

            # Create padding to make all images the same size
            padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            padded[:new_h, :new_w, :] = resized
            resized_images.append(padded)

        # Create empty mosaic
        mosaic_height = grid_rows * target_height
        mosaic_width = grid_cols * target_width
        mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)

        # Place images in mosaic
        for i, img in enumerate(resized_images):
            row = i // grid_cols
            col = i % grid_cols
            y_start = row * target_height
            x_start = col * target_width
            mosaic[y_start : y_start + target_height, x_start : x_start + target_width] = img

        return mosaic

    def _visualize(self):
        training_mode = self.model.training

        with ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            stack.enter_context(inference_context(self.model))
            stack.enter_context(inference_context(self.processor))

            storage = get_event_storage()
            self.model.eval()

            all_visualized_images = []

            for i in range(self.n_sample):
                sample = self.samples[i]
                sample["height"], sample["width"] = sample["image"].shape[-2:]

                samples = [sample]
                images, _ = self.processor.preprocess(samples, device=self.model.device, dtype=self.model.dtype)
                outputs = self.model(images)
                prediction = self.processor.eval_postprocess(outputs, samples)[0]

                visualizer = Visualizer(
                    sample["image"].permute(1, 2, 0).cpu().numpy(),
                    self.metadata,
                    instance_mode=ColorMode.IMAGE,
                )
                if "panoptic_seg" in prediction:
                    panoptic_seg, segments_info = prediction["panoptic_seg"]
                    vis_output = visualizer.draw_panoptic_seg_predictions(
                        panoptic_seg.to(self.cpu_device), segments_info
                    )
                elif "sem_seg" in prediction:
                    vis_output = visualizer.draw_sem_seg(prediction["sem_seg"].argmax(dim=0).to(self.cpu_device))
                elif "instances" in prediction:
                    instances = prediction["instances"].to(self.cpu_device)
                    # filter based on confidence - fixed at 0.5
                    instances = instances[instances.scores > 0.5]
                    vis_output = visualizer.draw_instance_predictions(predictions=instances)
                else:
                    vis_output = None

                if vis_output is not None:
                    pred_img = vis_output.get_image()
                    # Non salviamo più i singoli samples nello storage
                    all_visualized_images.append(pred_img)

            # Create and save mosaic if we have images and output directory
            if all_visualized_images:
                # Get current iteration for filename
                try:
                    current_iter = self.trainer.iter
                except (AttributeError, TypeError):
                    current_iter = 0

                # Create mosaic
                mosaic = self._create_mosaic(all_visualized_images)

                if mosaic is not None:
                    # Salva il mosaico nello storage invece dei singoli samples
                    mosaic_transposed = mosaic.transpose(2, 0, 1)  # HWC -> CHW
                    storage.put_image("Samples_Mosaic", mosaic_transposed)

                    # Save to disk if output_dir is provided
                    if self.output_dir is not None:
                        preview_dir = os.path.join(self.output_dir, "preview")
                        os.makedirs(preview_dir, exist_ok=True)

                        # Include iteration in filename
                        output_path = os.path.join(preview_dir, f"samples_iter_{current_iter}.jpg")
                        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
                        cv2.imwrite(output_path, mosaic, encode_params)

        # set model back to training mode
        self.model.train(training_mode)

    @property
    def iter(self):
        try:
            return self.trainer.iter
        except Exception:
            return 0

    def after_step(self):
        next_iter = self.iter + 1
        if self._period > 0 and next_iter % self._period == 0:
            # do the last eval in after_train
            if next_iter != self.trainer.max_iter:
                self._visualize()

    def after_train(self):
        try:
            # This condition is to prevent the eval from running after a failed training
            if self.trainer.max_iter is not None and self.iter + 1 >= self.trainer.max_iter:
                self._visualize()
        except (AttributeError, TypeError):
            # In case self.trainer is None
            self._visualize()
