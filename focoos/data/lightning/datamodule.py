"""
Lightning DataModule for Focoos.
DataModule completo con setup automatico, augmentations e funzione sample().
"""

import os
from typing import Any, Dict, Optional

import albumentations as A
import cv2
import lightning as L
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from focoos.data.datasets.dict_dataset import DictDataset
from focoos.ports import DATASETS_DIR, DatasetLayout, DatasetSplitType, Task
from focoos.structures import BoxMode
from focoos.utils.logger import get_logger
from focoos.utils.system import extract_archive

from .dataset import LightningDatasetWrapper
from .default_augmentations import get_default_train_augmentations, get_default_val_augmentations


class FocoosLightningDataModule(L.LightningDataModule):
    """
    DataModule Lightning semplificato che usa direttamente DictDataset di Focoos.
    Architettura piatta: DictDataset → LightningDatasetWrapper → DataLoader
    Setup automatico nell'__init__ - pronto all'uso immediato.
    """

    dataset_name: str
    datasets_dir: str
    task: Task
    layout: DatasetLayout
    dataset_path: str
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    image_size: int
    train_augmentations: A.Compose
    val_augmentations: A.Compose
    logger_focoos: Any
    train_dataset: LightningDatasetWrapper
    val_dataset: LightningDatasetWrapper

    def __init__(
        self,
        dataset_name: str,
        task: Task,
        layout: DatasetLayout,
        datasets_dir: str = DATASETS_DIR,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        train_augmentations: Optional[A.Compose] = None,
        val_augmentations: Optional[A.Compose] = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__()

        # Parametri base
        self.dataset_name = dataset_name
        self.datasets_dir = datasets_dir
        self.task = task
        self.layout = layout

        # Initialize logger early
        self.logger_focoos = get_logger("FocoosLightningDataModule")

        self.dataset_path = (
            datasets_dir if layout == DatasetLayout.CATALOG else os.path.join(datasets_dir, dataset_name)
        )
        if self.dataset_path.endswith(".zip") or self.dataset_path.endswith(".gz"):
            _d_name = dataset_name.split(".")[0]
            _dest_path = os.path.join(self.datasets_dir, _d_name)
            if not os.path.exists(_dest_path):
                self.logger_focoos.info(f"📦 Extracting dataset from {self.dataset_path}...")
                self.dataset_path = str(extract_archive(self.dataset_path))
                self.logger_focoos.info(f"Extracted archive: {self.dataset_path}, {os.listdir(self.dataset_path)}")
            else:
                self.logger_focoos.info(f"Dataset already extracted: {_dest_path}")
                self.dataset_path = _dest_path

        # Parametri DataLoader
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0

        # Augmentations
        self.image_size = image_size
        self.train_augmentations = train_augmentations or get_default_train_augmentations(self.task, self.image_size)
        self.val_augmentations = val_augmentations or get_default_val_augmentations(self.task, self.image_size)

        # Save hyperparameters and log augmentations
        self.save_hyperparameters()
        self._log_augmentations()

        # Setup automatico
        self._setup()

    def _log_augmentations(self) -> None:
        """Log delle augmentations per train e val, ispirato ai DatasetMapper di Focoos"""

        def format_augmentations(augs: Optional[A.Compose]) -> str:
            """Formatta le augmentations in una stringa leggibile"""
            if augs is None:
                return "None"

            aug_list = []
            if hasattr(augs, "transforms"):
                for aug in augs.transforms:
                    aug_name = aug.__class__.__name__
                    params = []
                    if hasattr(aug, "p") and aug.p != 1.0:
                        params.append(f"p={aug.p}")

                    if aug_name in ["Resize", "LetterBox"]:
                        height = getattr(aug, "height", None)
                        width = getattr(aug, "width", None)
                        if height and width:
                            params.append(f"size=({height}x{width})")
                    elif aug_name == "Normalize":
                        mean = getattr(aug, "mean", None)
                        std = getattr(aug, "std", None)
                        if mean and std:
                            mean_str = str(mean[:3]) if len(mean) > 3 else str(mean)
                            std_str = str(std[:3]) if len(std) > 3 else str(std)
                            params.append(f"mean={mean_str}")
                            params.append(f"std={std_str}")

                    param_str = f"({', '.join(params)})" if params else ""
                    aug_list.append(f"{aug_name}{param_str}")

            return "\n - ".join(aug_list)

        train_augs_str = format_augmentations(self.train_augmentations)
        self.logger_focoos.info(
            f"\n =========== 🎨 train augmentations =========== \n - {train_augs_str} \n============================================"
        )

        val_augs_str = format_augmentations(self.val_augmentations)
        self.logger_focoos.info(
            f"\n =========== 🎨 val augmentations =========== \n - {val_augs_str} \n============================================"
        )

    def _load_dict_dataset(self, split: DatasetSplitType) -> DictDataset:
        """Carica DictDataset direttamente in base al layout"""

        if split == DatasetSplitType.TRAIN:
            split_names = ["train", "training"]
        else:  # VAL
            split_names = ["valid", "val", "validation"]

        if self.layout != DatasetLayout.CATALOG:
            split_dir = None
            for name in split_names:
                candidate = os.path.join(self.dataset_path, name)
                if os.path.exists(candidate):
                    split_dir = candidate
                    break

            if split_dir is None:
                raise FileNotFoundError(
                    f"Split directory not found in {self.dataset_path} . Tried: {', '.join(split_names)}"
                )

        if self.layout == DatasetLayout.ROBOFLOW_COCO:
            return DictDataset.from_roboflow_coco(ds_dir=split_dir, task=self.task, split_type=split)
        elif self.layout == DatasetLayout.CLS_FOLDER:
            return DictDataset.from_folder(root_dir=split_dir, split_type=split)
        elif self.layout == DatasetLayout.ROBOFLOW_SEG:
            return DictDataset.from_roboflow_seg(ds_dir=split_dir, task=self.task, split_type=split)
        elif self.layout == DatasetLayout.CATALOG:
            return DictDataset.from_catalog(ds_name=self.dataset_name, split_type=split, root=self.datasets_dir)
        else:
            raise ValueError(f"Layout {self.layout} non supportato")

    def _setup(self) -> None:
        """Setup automatico dei dataset per training e validation"""
        train_dict_dataset = self._load_dict_dataset(DatasetSplitType.TRAIN)
        val_dict_dataset = self._load_dict_dataset(DatasetSplitType.VAL)

        self.train_dataset = LightningDatasetWrapper(
            dict_dataset=train_dict_dataset,
            transform=self.train_augmentations,
            use_mosaic=True,
            mosaic_p=0.5,
        )

        self.val_dataset = LightningDatasetWrapper(
            dict_dataset=val_dict_dataset,
            transform=self.val_augmentations,
            use_mosaic=False,
        )

        self.logger_focoos.info(
            f"✅ Setup completato - Train: {len(self.train_dataset)} samples, Val: {len(self.val_dataset)} samples"
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Compatibilità con Lightning Trainer (setup già eseguito in __init__)"""
        pass

    def _collate_fn(self, batch):
        """Custom collate function - batch is already a list of DatasetEntry objects"""
        # No conversion needed - LightningDatasetWrapper already returns DatasetEntry
        return batch

    def train_dataloader(self):
        """Ritorna il DataLoader per il training"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        """Ritorna il DataLoader per la validation"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=self._collate_fn,
        )

    def _draw_annotations(
        self,
        image: np.ndarray,
        bboxes: list,
        labels: list,
        class_names: Optional[list],
        index: int,
    ) -> np.ndarray:
        """
        Disegna bounding boxes, labels e indice su un'immagine usando OpenCV.

        Args:
            image: Immagine numpy array in formato RGB
            bboxes: Lista di bounding boxes [x_min, y_min, x_max, y_max]
            labels: Lista di labels corrispondenti
            class_names: Lista dei nomi delle classi (opzionale)
            index: Indice del sample da visualizzare

        Returns:
            Immagine annotata (modifica in-place ma ritorna per chiarezza)
        """
        # Disegna bboxes
        for bbox, label in zip(bboxes, labels):
            x_min, y_min, x_max, y_max = map(int, bbox)
            # Disegna rettangolo
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Rosso in RGB

            # Disegna label
            class_name = class_names[int(label)] if class_names and label < len(class_names) else f"C{int(label)}"
            # Calcola dimensioni del testo
            (text_width, text_height), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # Disegna sfondo del testo
            cv2.rectangle(image, (x_min, y_min - text_height - 5), (x_min + text_width, y_min), (255, 0, 0), -1)
            # Disegna testo
            cv2.putText(image, class_name, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Disegna l'indice
        index_text = f"#{index}"
        (text_width, text_height), baseline = cv2.getTextSize(index_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # Sfondo blu
        cv2.rectangle(image, (5, 5), (8 + text_width, 8 + text_height), (0, 0, 255), -1)  # Blu in RGB
        # Testo bianco
        cv2.putText(image, index_text, (5, 5 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return image

    def sample(
        self,
        split: str = "train",
        index: Optional[int] = None,
        with_augmentations: bool = True,
        random_seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Ritorna una singola immagine dal dataset con opzione di mostrare le augmentations.
        L'immagine include l'indice visualizzato nell'angolo in alto a sinistra.

        Args:
            split: "train" o "val" per scegliere lo split
            index: Indice specifico del sample (se None, sceglie casualmente)
            with_augmentations: Se True, applica augmentations; se False, solo resize
            random_seed: Seed per la riproducibilità quando index è None

        Returns:
            PIL.Image: Immagine con annotazioni disegnate (bounding boxes per detection, indice, etc.)
        """
        if not hasattr(self, "train_dataset"):
            raise RuntimeError("Devi chiamare setup() prima di usare sample()")

        dataset = self.train_dataset if split == "train" else self.val_dataset if split == "val" else None
        if dataset is None:
            raise ValueError(f"Split deve essere 'train' o 'val', non {split}")

        # Seleziona indice
        if index is None:
            if random_seed is not None:
                np.random.seed(random_seed)
            index = np.random.randint(0, len(dataset))

        # Se non vogliamo augmentations, carica direttamente con OpenCV senza passare per tensor
        if not with_augmentations:
            item = dataset.dict_dataset[index]
            # Carica con OpenCV e converti in RGB (cv2 carica in BGR)
            image = cv2.imread(item["file_name"])
            if image is None:
                raise ValueError(f"Impossibile caricare l'immagine: {item['file_name']}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Per detection, estrai e disegna bboxes
            bboxes_list = []
            labels_list = []

            if self.task in [Task.DETECTION, Task.INSTANCE_SEGMENTATION]:
                if "annotations" in item and len(item["annotations"]) > 0:
                    for ann in item["annotations"]:
                        bbox = BoxMode.convert(ann["bbox"], ann["bbox_mode"], BoxMode.XYXY_ABS)
                        bboxes_list.append(bbox)
                        labels_list.append(ann["category_id"])

            # Disegna annotazioni (bboxes + indice)
            class_names = getattr(dataset.dict_dataset.metadata, "thing_classes", None)
            image = self._draw_annotations(image, bboxes_list, labels_list, class_names, index)

            # Converti in PIL per il return
            pil_image = Image.fromarray(image)
            sample = None
        else:
            # Con augmentations, usa il pipeline normale
            sample = dataset[index]

            if sample.image is None:
                raise ValueError(f"Image tensor is None for sample {index}")

            image_tensor = sample.image
            image = image_tensor.cpu().numpy()
            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)

            # Controlla se il transform include normalizzazione
            has_normalize = False
            if hasattr(dataset, "transform") and dataset.transform is not None:
                if hasattr(dataset.transform, "transforms"):
                    for t in dataset.transform.transforms:
                        if t.__class__.__name__ == "Normalize":
                            has_normalize = True
                            break

            # Denormalizza solo se l'immagine è stata normalizzata
            if has_normalize:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = std * image + mean

            image = np.clip(image, 0, 1)
            # Converti in uint8 per OpenCV
            image = (image * 255).astype(np.uint8)

            # Estrai bboxes e labels se disponibili
            bboxes_list = []
            labels_list = []
            if hasattr(sample, "instances") and sample.instances is not None:
                if sample.instances.boxes is not None:
                    bboxes_list = sample.instances.boxes.tensor.cpu().numpy().tolist()
                if sample.instances.classes is not None:
                    labels_list = sample.instances.classes.cpu().numpy().tolist()

            # Disegna annotazioni con OpenCV
            class_names = getattr(dataset.dict_dataset.metadata, "thing_classes", None)
            image = self._draw_annotations(image, bboxes_list, labels_list, class_names, index)

            # Converti in PIL per il return
            pil_image = Image.fromarray(image)

        return pil_image

    def get_dataset_info(self) -> Dict[str, Any]:
        """Ritorna informazioni sul dataset (compatibile con AutoDataset)"""
        info: Dict[str, Any] = {
            "dataset_name": self.dataset_name,
            "datasets_dir": self.datasets_dir,
            "task": self.task.value,
            "layout": self.layout.value,
        }

        if hasattr(self, "train_dataset"):
            info["train_samples"] = len(self.train_dataset)
            info["val_samples"] = len(self.val_dataset) if hasattr(self, "val_dataset") else 0
            info["num_classes"] = self.train_dataset.dict_dataset.metadata.num_classes
            info["class_names"] = getattr(self.train_dataset.dict_dataset.metadata, "thing_classes", None)

        return info
