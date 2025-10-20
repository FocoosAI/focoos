"""
Lightning Dataset Wrapper for Focoos.
Wrapper che integra DictDataset con Albumentations e supporto Mosaic.
"""

from typing import Any, Dict, List, Optional, Union

import albumentations as A
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import Dataset

from focoos.data.datasets.dict_dataset import DictDataset
from focoos.ports import Task
from focoos.structures import BoxMode

from .types import ClassificationSample, DetectionSample, GenericSample, SegmentationSample


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
        use_mosaic: bool = False,
        mosaic_p: float = 0.5,
    ) -> None:
        """
        Args:
            dict_dataset: DictDataset di Focoos
            transform: Albumentations transform pipeline
            use_mosaic: Se True, applica Mosaic augmentation (solo per detection tasks)
            mosaic_p: Probabilità di applicare Mosaic (default: 0.5)
        """
        self.dict_dataset = dict_dataset
        self.transform = transform
        self.task = dict_dataset.metadata.task
        self.metadata = dict_dataset.metadata
        self.use_mosaic = use_mosaic and self.task in [Task.DETECTION, Task.INSTANCE_SEGMENTATION]
        self.mosaic_p = mosaic_p

    def __len__(self) -> int:
        return len(self.dict_dataset)

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

    def __getitem__(self, idx: int) -> Union[DetectionSample, ClassificationSample, SegmentationSample, GenericSample]:
        item: Dict[str, Any] = self.dict_dataset[idx]
        image: NDArray[np.uint8] = np.array(Image.open(item["file_name"]).convert("RGB"))

        if self.task in [Task.DETECTION, Task.INSTANCE_SEGMENTATION]:
            return self._process_detection(image, item, idx)
        elif self.task == Task.CLASSIFICATION:
            return self._process_classification(image, item, idx)
        elif self.task == Task.SEMSEG:
            return self._process_segmentation(image, item, idx)
        else:
            return self._process_generic(image, item, idx)

    def _process_detection(self, image: NDArray[np.uint8], item: Dict[str, Any], idx: int) -> DetectionSample:
        """Processa sample per detection/instance segmentation con supporto Mosaic"""
        bboxes: List[Any] = []
        labels: List[Any] = []

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

        # Denormalizza bboxes
        if len(bboxes) > 0:
            _, h, w = image_tensor.shape
            bboxes_tensor = torch.tensor(
                [[b[0] * w, b[1] * h, b[2] * w, b[3] * h] for b in bboxes], dtype=torch.float32
            )
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            bboxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros(0, dtype=torch.int64)

        return DetectionSample(
            image=image_tensor,
            bboxes=bboxes_tensor,
            labels=labels_tensor,
            image_id=item.get("image_id", idx),
            file_name=item["file_name"],
        )

    def _process_classification(self, image: NDArray[np.uint8], item: Dict[str, Any], idx: int) -> ClassificationSample:
        """Processa sample per classificazione"""
        label = item["annotations"][0]["category_id"] if "annotations" in item and len(item["annotations"]) > 0 else -1

        image_tensor = (
            self.transform(image=image)["image"]
            if self.transform
            else torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        )

        return ClassificationSample(
            image=image_tensor, label=label, image_id=item.get("image_id", idx), file_name=item["file_name"]
        )

    def _process_segmentation(self, image: NDArray[np.uint8], item: Dict[str, Any], idx: int) -> SegmentationSample:
        """Processa sample per semantic segmentation"""
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
            mask_tensor = torch.from_numpy(transformed["mask"]).long()
        else:
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(mask).long()

        return SegmentationSample(
            image=image_tensor,
            mask=mask_tensor,
            image_id=item.get("image_id", idx),
            file_name=item["file_name"],
            sem_seg_file_name=sem_seg_file_name,
        )

    def _process_generic(self, image: NDArray[np.uint8], item: Dict[str, Any], idx: int) -> GenericSample:
        """Processa sample per task generici"""
        image_tensor = (
            self.transform(image=image)["image"]
            if self.transform
            else torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        )

        return GenericSample(image=image_tensor, image_id=item.get("image_id", idx), file_name=item["file_name"])
