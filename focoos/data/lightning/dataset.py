"""
Lightning Dataset Wrapper for Focoos.
Wrapper che integra DictDataset con Albumentations e supporto Mosaic.
"""

from typing import Any, Dict, List, Optional

import albumentations as A
import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import Dataset

from focoos.data.datasets.dict_dataset import DictDataset
from focoos.ports import DatasetEntry, Task
from focoos.structures import BitMasks, Boxes, BoxMode, Instances


class LightningDatasetWrapper(Dataset):
    """
    Wrapper semplificato per DictDataset di Focoos con supporto Albumentations.
    Il core è il DictDataset che contiene tutta la logica per caricare dataset COCO.
    """

    dict_dataset: DictDataset
    transform: Optional[A.Compose]
    task: Task
    metadata: Any
    use_mosaic: bool
    mosaic_p: float

    def __init__(
        self,
        dict_dataset: DictDataset,
        transform: Optional[A.Compose] = None,
    ) -> None:
        """
        Args:
            dict_dataset: DictDataset di Focoos
            transform: Albumentations transform pipeline
        """
        self.dict_dataset = dict_dataset
        self.transform = transform
        self.task = dict_dataset.metadata.task
        self.metadata = dict_dataset.metadata

        # Auto-detect Mosaic augmentation from transform pipeline
        self.use_mosaic = False
        self.mosaic_p = 0.0
        self._detect_mosaic_augmentation()

    def __len__(self) -> int:
        return len(self.dict_dataset)

    def _detect_mosaic_augmentation(self) -> None:
        """
        Auto-detect if Mosaic augmentation is present in the transform pipeline.
        Sets self.use_mosaic and self.mosaic_p accordingly.
        """
        if not self.transform or self.task not in [Task.DETECTION, Task.INSTANCE_SEGMENTATION]:
            return

        # Check if transform has a transforms attribute (A.Compose)
        if not hasattr(self.transform, "transforms"):
            return

        # Look for Mosaic augmentation in the pipeline
        for aug in self.transform.transforms:
            aug_name = aug.__class__.__name__
            if aug_name == "Mosaic":
                # Extract probability from the augmentation
                if hasattr(aug, "p"):
                    self.mosaic_p = aug.p
                self.use_mosaic = True if aug.p > 0.0 else False
                break

    def _load_additional_images(self, current_idx: int, num_images: int = 3) -> List[Dict[str, Any]]:
        """
        Carica immagini aggiuntive random dal dataset per Mosaic augmentation.

        Args:
            current_idx: Indice dell'immagine corrente (da escludere)
            num_images: Numero di immagini aggiuntive da caricare

        Returns:
            Lista di dict con 'image', 'bboxes', 'labels'
        """
        additional_images = []

        # Seleziona indici random escludendo quello corrente
        available_indices = list(range(len(self.dict_dataset)))
        if current_idx in available_indices:
            available_indices.remove(current_idx)

        if len(available_indices) < num_images:
            # Non abbastanza immagini, ritorna lista vuota
            return []

        indices = np.random.choice(available_indices, size=num_images, replace=False)

        for idx in indices:
            item = self.dict_dataset[idx]
            add_image = np.array(Image.open(item["file_name"]).convert("RGB"))
            add_bboxes = []
            add_labels = []

            if "annotations" in item and len(item["annotations"]) > 0:
                h, w = add_image.shape[:2]
                for ann in item["annotations"]:
                    bbox = BoxMode.convert(ann["bbox"], ann["bbox_mode"], BoxMode.XYXY_ABS)
                    x_min, y_min, x_max, y_max = bbox
                    add_bboxes.append([x_min / w, y_min / h, x_max / w, y_max / h])
                    add_labels.append(ann["category_id"])

            # Converti in numpy arrays per Albumentations
            add_bboxes_array = (
                np.array(add_bboxes, dtype=np.float32) if len(add_bboxes) > 0 else np.zeros((0, 4), dtype=np.float32)
            )
            add_labels_array = (
                np.array(add_labels, dtype=np.int64) if len(add_labels) > 0 else np.zeros(0, dtype=np.int64)
            )

            additional_images.append({"image": add_image, "bboxes": add_bboxes_array, "labels": add_labels_array})

        return additional_images

    def __getitem__(self, idx: int) -> DatasetEntry:
        item: Dict[str, Any] = self.dict_dataset[idx]
        image: NDArray[np.uint8] = cv2.cvtColor(cv2.imread(item["file_name"]), cv2.COLOR_BGR2RGB)

        if self.task in [Task.DETECTION, Task.INSTANCE_SEGMENTATION]:
            return self._process_detection(image, item, idx)
        elif self.task == Task.CLASSIFICATION:
            return self._process_classification(image, item, idx)
        elif self.task == Task.SEMSEG:
            return self._process_segmentation(image, item, idx)
        else:
            return self._process_generic(image, item, idx)

    def _process_detection(self, image: NDArray[np.uint8], item: Dict[str, Any], idx: int) -> DatasetEntry:
        """Processa sample per detection/instance segmentation con supporto Mosaic"""
        bboxes: List[Any] = []
        labels: List[Any] = []

        # IMPORTANTE: Salva dimensioni originali PRIMA delle augmentations
        # Queste saranno usate dal processor per scalare le predizioni alle dimensioni target
        original_height, original_width = image.shape[:2]

        if "annotations" in item and len(item["annotations"]) > 0:
            h, w = image.shape[:2]
            for ann in item["annotations"]:
                bbox = BoxMode.convert(ann["bbox"], ann["bbox_mode"], BoxMode.XYXY_ABS)
                x_min, y_min, x_max, y_max = bbox
                bboxes.append([x_min / w, y_min / h, x_max / w, y_max / h])
                labels.append(ann["category_id"])

        # Applica transform (Mosaic viene gestito automaticamente se presente nel pipeline)
        if self.transform:
            # Converti bboxes e labels in numpy arrays per Albumentations
            bboxes_array = np.array(bboxes, dtype=np.float32) if len(bboxes) > 0 else np.zeros((0, 4), dtype=np.float32)
            labels_array = np.array(labels, dtype=np.int64) if len(labels) > 0 else np.zeros(0, dtype=np.int64)

            # Prepara i dati per il transform
            transform_data = {
                "image": image,
                "bboxes": bboxes_array,
                "labels": labels_array,
            }

            # Se Mosaic è abilitato, carica immagini aggiuntive e passa come metadata
            if self.use_mosaic:
                additional_images = self._load_additional_images(idx, num_images=3)
                if len(additional_images) >= 3:
                    transform_data["mosaic_metadata"] = additional_images

            # Applica tutte le transforms (incluso Mosaic se presente)
            transformed = self.transform(**transform_data)
            image_tensor = transformed["image"]
            bboxes = list(transformed["bboxes"])
            labels = list(transformed["labels"])
        else:
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Get dimensions of TRANSFORMED image (after augmentations)
        _, height_transformed, width_transformed = image_tensor.shape

        # Create DatasetEntry with ORIGINAL dimensions
        # Il processor userà queste dimensioni per scalare le predizioni alle dimensioni target originali
        entry = DatasetEntry(
            image=image_tensor,
            height=original_height,  # ← Dimensioni originali (prima delle augmentations)
            width=original_width,  # ← Dimensioni originali (prima delle augmentations)
            image_id=item.get("image_id", idx),
            file_name=item["file_name"],
        )

        # Always create Instances object for detection/segmentation (even if empty)
        # Denormalize bboxes to TRANSFORMED image dimensions (l'immagine che abbiamo effettivamente)
        if len(bboxes) > 0:
            # Convert to numpy first for efficient vectorized operations, then to tensor
            bboxes_array = np.array(bboxes, dtype=np.float32)
            bboxes_array[:, [0, 2]] *= width_transformed  # x coordinates sulle dimensioni trasformate
            bboxes_array[:, [1, 3]] *= height_transformed  # y coordinates sulle dimensioni trasformate
            bboxes_tensor = torch.from_numpy(bboxes_array)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        else:
            bboxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros(0, dtype=torch.int64)

        # Create Instances object with boxes and classes passed to constructor
        # image_size deve essere delle dimensioni trasformate (corrispondenti alle bbox)
        instances = Instances(
            image_size=(height_transformed, width_transformed),
            boxes=Boxes(bboxes_tensor),
            classes=labels_tensor,
        )
        entry.instances = instances

        return entry

    def _process_classification(self, image: NDArray[np.uint8], item: Dict[str, Any], idx: int) -> DatasetEntry:
        """Processa sample per classificazione"""
        # Salva dimensioni originali PRIMA delle augmentations
        original_height, original_width = image.shape[:2]

        label = item["annotations"][0]["category_id"] if "annotations" in item and len(item["annotations"]) > 0 else -1

        image_tensor = (
            self.transform(image=image)["image"]
            if self.transform
            else torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        )

        entry = DatasetEntry(
            image=image_tensor,
            height=original_height,  # ← Dimensioni originali
            width=original_width,  # ← Dimensioni originali
            image_id=item.get("image_id", idx),
            file_name=item["file_name"],
        )
        entry.label = label

        return entry

    def _process_segmentation(self, image: NDArray[np.uint8], item: Dict[str, Any], idx: int) -> DatasetEntry:
        """Processa sample per semantic segmentation"""
        # IMPORTANTE: Salva dimensioni originali PRIMA delle augmentations
        # Queste saranno usate dal processor per scalare le predizioni alle dimensioni target
        original_height, original_width = image.shape[:2]

        # Carica la maschera di segmentazione
        if "sem_seg_file_name" not in item:
            raise ValueError(f"sem_seg_file_name not found in item {idx}")

        sem_seg_file_name = item["sem_seg_file_name"]
        # Leggi la maschera come grayscale
        mask = np.array(Image.open(sem_seg_file_name))

        # Se la maschera è RGB, prendi solo il primo canale
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        # Applica transforms (Albumentations supporta mask come target)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image_tensor = transformed["image"]
            # Use as_tensor which handles both tensor and numpy without unnecessary copies
            mask_tensor = torch.as_tensor(transformed["mask"], dtype=torch.long)
        else:
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(mask).long()

        # Get dimensions of TRANSFORMED image (after augmentations)
        _, height_transformed, width_transformed = image_tensor.shape

        # Create DatasetEntry with ORIGINAL dimensions
        # Il processor userà queste dimensioni per scalare le predizioni alle dimensioni target originali
        entry = DatasetEntry(
            image=image_tensor,
            height=original_height,  # ← Dimensioni originali (prima delle augmentations)
            width=original_width,  # ← Dimensioni originali (prima delle augmentations)
            image_id=item.get("image_id", idx),
            file_name=item["file_name"],
        )
        entry.sem_seg = mask_tensor
        entry.sem_seg_file_name = sem_seg_file_name  # For evaluation

        # Also populate instances for models that expect it (like MaskFormer)
        # Convert semantic mask to instance masks (one per class)
        unique_classes = torch.unique(mask_tensor)
        # Remove background class (0) if present
        unique_classes = unique_classes[unique_classes > 0]

        if len(unique_classes) > 0:
            # Create binary masks for each class
            instance_masks = []
            instance_classes = []
            for cls in unique_classes:
                binary_mask = mask_tensor == cls
                instance_masks.append(binary_mask)
                instance_classes.append(cls.item())

            masks_tensor = torch.stack(instance_masks)  # Shape: (N, H, W)
            classes_tensor = torch.tensor(instance_classes, dtype=torch.int64)
        else:
            # Use transformed dimensions for empty masks
            masks_tensor = torch.zeros((0, height_transformed, width_transformed), dtype=torch.bool)
            classes_tensor = torch.zeros(0, dtype=torch.int64)

        # Create Instances with masks
        # image_size deve essere delle dimensioni trasformate (corrispondenti alle mask)
        instances = Instances(
            image_size=(height_transformed, width_transformed),
            masks=BitMasks(masks_tensor),
            classes=classes_tensor,
        )
        entry.instances = instances

        return entry

    def _process_generic(self, image: NDArray[np.uint8], item: Dict[str, Any], idx: int) -> DatasetEntry:
        """Processa sample per task generici"""
        # Salva dimensioni originali PRIMA delle augmentations
        original_height, original_width = image.shape[:2]

        image_tensor = (
            self.transform(image=image)["image"]
            if self.transform
            else torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        )

        return DatasetEntry(
            image=image_tensor,
            height=original_height,  # ← Dimensioni originali
            width=original_width,  # ← Dimensioni originali
            image_id=item.get("image_id", idx),
            file_name=item["file_name"],
        )
