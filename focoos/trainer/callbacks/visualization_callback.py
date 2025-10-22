# Copyright (c) FocoosAI
"""
Lightning callback for visualization during training.

This callback mimics the behavior of VisualizationHook from the original trainer,
creating visualizations of model predictions on sample images and saving them as mosaics.
"""

import math
import os
import random
from typing import List, Optional

import cv2
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

from focoos.utils.logger import get_logger
from focoos.utils.visualizer import ColorMode, Visualizer

logger = get_logger("VisualizationCallback")


class VisualizationCallback(Callback):
    """
    Lightning callback that visualizes model predictions during training.

    Creates side-by-side comparison images with ground truth annotations on the left
    and model predictions on the right. Saves images to disk and/or TensorBoard.
    Similar to VisualizationHook from the original trainer.

    Args:
        period: Visualize every period steps
        n_sample: Number of samples to visualize (default: 4)
        index_list: List of specific indices to visualize. If None, uses first n_sample or random
        random_samples: If True and index_list is None, randomly sample indices
        output_dir: Directory to save visualization images. If None, only logs to TensorBoard
        confidence_threshold: Minimum confidence score to display predictions (default: 0.5)

    Example:
        >>> viz_callback = VisualizationCallback(
        ...     period=200,
        ...     n_sample=4,
        ...     output_dir="output/visualizations",
        ... )
        >>> trainer = L.Trainer(callbacks=[viz_callback, ...])

    The callback will save images as: {output_dir}/preview/sample_{idx}_iter_{step}.jpg
    Each image contains ground truth (left) and predictions (right) side-by-side.
    """

    def __init__(
        self,
        period: int,
        n_sample: int = 2,
        index_list: Optional[List[int]] = None,
        random_samples: bool = False,
        output_dir: Optional[str] = None,
        confidence_threshold: float = 0.5,
    ):
        super().__init__()
        self.period = period
        self.n_sample = n_sample
        self.index_list = index_list
        self.random_samples = random_samples
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold

        # Will be set in setup
        self.samples = None
        self.metadata = None
        self.cpu_device = torch.device("cpu")

        logger.info(
            f"VisualizationCallback initialized: period={period}, n_sample={n_sample}, "
            f"output_dir={output_dir}, confidence_threshold={confidence_threshold}"
        )

    def setup(self, trainer, pl_module, stage: Optional[str] = None):
        """Setup callback - extract samples from validation dataset."""
        if stage == "fit" or stage is None:
            # Get validation dataset from datamodule
            datamodule = trainer.datamodule  # type: ignore
            if not hasattr(datamodule, "val_dataset"):
                logger.warning("No validation dataset found, skipping visualization setup")
                return

            val_dataset = datamodule.val_dataset
            dataset_len = len(val_dataset)

            # Select indices
            if self.index_list is None or len(self.index_list) <= 0:
                if self.random_samples:
                    self.index_list = random.sample(range(dataset_len), k=min(dataset_len, self.n_sample))
                else:
                    self.index_list = [i for i in range(min(dataset_len, self.n_sample))]

            self.n_sample = len(self.index_list)

            # Extract samples
            self.samples = [val_dataset[i] for i in self.index_list]

            # Get metadata from the underlying DictDataset
            if hasattr(val_dataset, "dict_dataset"):
                self.metadata = val_dataset.dict_dataset.metadata
            elif hasattr(val_dataset, "metadata"):
                self.metadata = val_dataset.metadata
            else:
                logger.warning("Could not extract metadata from validation dataset")
                self.metadata = None

            logger.info(f"Visualization samples prepared: {self.n_sample} samples selected")

    def _create_mosaic(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Create a mosaic of images with max resolution 3000x3000.

        NOTE: This method is currently not used. Images are saved individually.
        Kept for potential future use if mosaic visualization is needed again.

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

            # Ensure images are in RGB before resizing
            if len(img.shape) == 3 and img.shape[2] == 3:
                # Check if it's BGR (OpenCV format) and convert to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img

            resized = cv2.resize(img_rgb, (new_w, new_h))

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

    def _visualize(self, trainer, pl_module):
        """
        Generate visualizations of model predictions.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module containing the model
        """
        if self.samples is None or self.metadata is None:
            logger.warning("Samples or metadata not initialized, skipping visualization")
            return

        # Get model and processor from the Lightning module
        model = pl_module.model
        processor = pl_module.processor

        # Save training state
        training_mode = model.training
        processor_mode = processor.training

        try:
            # Set to eval mode
            model.eval()
            processor.eval()

            all_visualized_images = []

            with torch.no_grad():
                for i in range(self.n_sample):
                    sample = self.samples[i]

                    # Ensure height and width are set
                    if sample.image is not None:
                        sample.height, sample.width = sample.image.shape[-2:]

                    # Prepare batch
                    samples = [sample]
                    images, _ = processor.preprocess(samples, device=model.device, dtype=model.dtype)

                    # Run inference
                    outputs = model(images)

                    # Post-process predictions
                    prediction = processor.eval_postprocess(outputs, samples)[0]

                    # Convert image tensor to numpy array for visualization
                    # Images are in [0, 255] range from ToTensor transform
                    image_vis = sample.image.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC

                    # Convert to uint8 [0, 255] range
                    image_vis = np.clip(image_vis, 0, 255).astype(np.uint8)

                    # Create visualizer for ground truth
                    visualizer_gt = Visualizer(
                        image_vis.copy(),
                        self.metadata,
                        instance_mode=ColorMode.IMAGE,
                    )

                    # Draw ground truth annotations
                    gt_img = None
                    if hasattr(sample, "instances") and sample.instances is not None:
                        gt_instances = sample.instances.to(self.cpu_device)
                        vis_output_gt = visualizer_gt.draw_instance_predictions(predictions=gt_instances)
                        gt_img = vis_output_gt.get_image()
                    elif hasattr(sample, "sem_seg") and sample.sem_seg is not None:
                        vis_output_gt = visualizer_gt.draw_sem_seg(sample.sem_seg.to(self.cpu_device))
                        gt_img = vis_output_gt.get_image()

                    # Create visualizer for predictions
                    visualizer_pred = Visualizer(
                        image_vis.copy(),
                        self.metadata,
                        instance_mode=ColorMode.IMAGE,
                    )

                    # Draw predictions based on task
                    pred_img = None
                    if "panoptic_seg" in prediction:
                        panoptic_seg, segments_info = prediction["panoptic_seg"]
                        vis_output = visualizer_pred.draw_panoptic_seg_predictions(
                            panoptic_seg.to(self.cpu_device), segments_info
                        )
                        pred_img = vis_output.get_image()
                    elif "sem_seg" in prediction:
                        vis_output = visualizer_pred.draw_sem_seg(
                            prediction["sem_seg"].argmax(dim=0).to(self.cpu_device)
                        )
                        pred_img = vis_output.get_image()
                    elif "instances" in prediction:
                        instances = prediction["instances"].to(self.cpu_device)
                        # Filter based on confidence threshold
                        instances = instances[instances.scores > self.confidence_threshold]
                        vis_output = visualizer_pred.draw_instance_predictions(predictions=instances)
                        pred_img = vis_output.get_image()

                    # Concatenate GT (left) and Predictions (right) horizontally
                    if gt_img is not None and pred_img is not None:
                        # Ensure both images have the same height
                        h_gt, w_gt = gt_img.shape[:2]
                        h_pred, w_pred = pred_img.shape[:2]

                        if h_gt != h_pred:
                            # Resize to match heights
                            target_h = max(h_gt, h_pred)
                            if h_gt < target_h:
                                gt_img = cv2.resize(gt_img, (w_gt, target_h))
                            if h_pred < target_h:
                                pred_img = cv2.resize(pred_img, (w_pred, target_h))

                        # Add text labels to identify GT and Predictions
                        h_gt, w_gt = gt_img.shape[:2]
                        h_pred, w_pred = pred_img.shape[:2]

                        # Add "Ground Truth" label on GT image
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1.0
                        thickness = 2
                        text_gt = "Ground Truth"
                        text_pred = f"Predictions (conf > {self.confidence_threshold})"

                        # Get text size to create background rectangle
                        (text_w_gt, text_h_gt), _ = cv2.getTextSize(text_gt, font, font_scale, thickness)
                        (text_w_pred, text_h_pred), _ = cv2.getTextSize(text_pred, font, font_scale, thickness)

                        # Add background rectangle and text for GT
                        cv2.rectangle(gt_img, (10, 10), (20 + text_w_gt, 20 + text_h_gt), (0, 0, 0), -1)
                        cv2.putText(gt_img, text_gt, (15, 15 + text_h_gt), font, font_scale, (255, 255, 255), thickness)

                        # Add background rectangle and text for Predictions
                        cv2.rectangle(pred_img, (10, 10), (20 + text_w_pred, 20 + text_h_pred), (0, 0, 0), -1)
                        cv2.putText(
                            pred_img,
                            text_pred,
                            (15, 15 + text_h_pred),
                            font,
                            font_scale,
                            (255, 255, 255),
                            thickness,
                        )

                        # Concatenate horizontally: GT on left, predictions on right
                        combined_img = np.hstack([gt_img, pred_img])
                        all_visualized_images.append(combined_img)

            # Save individual images if we have images
            if all_visualized_images:
                current_step = trainer.global_step

                # Save to disk if output_dir is provided
                if self.output_dir is not None:
                    preview_dir = os.path.join(self.output_dir, "preview")
                    os.makedirs(preview_dir, exist_ok=True)

                    # Save each sample as a separate file
                    for idx, img in enumerate(all_visualized_images):
                        output_path = os.path.join(preview_dir, f"sample_{idx}_iter_{current_step}.jpg")
                        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
                        cv2.imwrite(output_path, img, encode_params)

                    logger.info(f"Saved {len(all_visualized_images)} visualization samples to: {preview_dir}")

                # Log to TensorBoard (log all individual images)
                if trainer.logger is not None:
                    try:
                        if hasattr(trainer.logger, "experiment"):
                            for idx, img in enumerate(all_visualized_images):
                                # Convert to CHW format for TensorBoard
                                img_transposed = img.transpose(2, 0, 1)  # HWC -> CHW
                                trainer.logger.experiment.add_image(
                                    f"Samples/sample_{idx}",
                                    img_transposed,
                                    global_step=current_step,
                                )
                    except Exception as e:
                        logger.warning(f"Could not log to TensorBoard: {e}")

        except Exception as e:
            logger.warning(f"Exception during visualization: {e}", exc_info=True)

        finally:
            # Restore training mode
            model.train(training_mode)
            processor.train() if processor_mode else processor.eval()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when a training batch ends - check if we should visualize."""
        if self.period <= 0:
            return

        next_step = trainer.global_step + 1

        # Visualize every period steps (but not on the last step, which is handled in on_train_end)
        if next_step % self.period == 0:
            # Get max steps/iterations
            max_steps = trainer.max_steps if trainer.max_steps > 0 else float("inf")

            # Don't visualize on the last step (will be done in on_train_end)
            if next_step < max_steps:
                self._visualize(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        """Called when training ends - create final visualization."""
        try:
            # Create final visualization
            self._visualize(trainer, pl_module)
            logger.info("Final visualization created at end of training")
        except Exception as e:
            logger.warning(f"Could not create final visualization: {e}")
