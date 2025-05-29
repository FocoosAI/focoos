import random

import numpy as np
import supervision as sv
import torch.utils.data as data
from PIL import Image

from focoos.data.datasets.dict_dataset import DictDataset
from focoos.data.mappers.mapper import DatasetMapper
from focoos.ports import Task
from focoos.utils.logger import get_logger


class MapDataset(data.Dataset):
    """
    Map a function over the elements in a dataset.
    """

    def __init__(self, dataset: DictDataset, mapper: DatasetMapper):
        """
        Args:
            dataset: a dataset where map function is applied. Can be either
                map-style or iterable dataset. When given an iterable dataset,
                the returned object will also be an iterable dataset.
            map_func: a callable which maps the element in dataset. map_func can
                return None to skip the data (e.g. in case of errors).
                How None is handled depends on the style of `dataset`.
                If `dataset` is map-style, it randomly tries other elements.
                If `dataset` is iterable, it skips the data and tries the next.
        """
        self.dataset = dataset
        self.mapper = mapper  # wrap so that a lambda will work
        self.logger = get_logger(__name__)

        self._rng = random.Random(42)
        self._fallback_candidates = set(range(len(dataset)))

    def __getnewargs__(self):
        return self.dataset, self.mapper

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)

        while True:
            try:
                data = self.mapper(self.dataset[cur_idx])
            except Exception as e:
                self.logger.warning(f"Error mapping item {cur_idx}: {e}")
                data = None
                if retry_count >= 10:
                    raise e

            if data is not None and (data.instances is None or len(data.instances) > 0):
                # if it has annotations, it must more than 1 instance, otherwise it is not a valid training data
                self._fallback_candidates.add(cur_idx)
                return data

            # _map_func fails for this idx, use a random new index from the pool
            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(sorted(self._fallback_candidates), k=1)[0]

            if retry_count >= 3:
                self.logger.warning("Failed to apply `_map_func` for idx: {}, retry count: {}".format(idx, retry_count))

    def show_sample_image(self, index=None):
        index = index or random.randint(0, len(self.dataset))
        task = self.dataset.metadata.task
        classes = self.dataset.metadata.classes

        label_annotator = sv.LabelAnnotator(text_padding=10, border_radius=10)
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        sample = self[index]

        im = np.array(sample.image).transpose(1, 2, 0)

        sv_detections = sv.Detections(
            xyxy=sample["instances"].boxes.tensor.numpy(),
            class_id=sample["instances"].classes.numpy(),
            confidence=np.ones_like(sample["instances"].classes.numpy()),
            mask=sample["instances"].masks.tensor.numpy()
            if task in [Task.INSTANCE_SEGMENTATION, Task.SEMSEG]
            else None,
        )

        if len(sv_detections.xyxy) == 0:
            print("No detections found, skipping annotation")
            return Image.fromarray(im)

        if task == Task.DETECTION:
            annotated_im = box_annotator.annotate(scene=im.copy(), detections=sv_detections)

        elif task in [
            Task.SEMSEG,
            Task.INSTANCE_SEGMENTATION,
        ]:
            annotated_im = mask_annotator.annotate(scene=im.copy(), detections=sv_detections)

        # Fixme: get the classes from the detections
        if classes is not None:
            labels = [
                f"{classes[int(class_id)] if classes is not None else str(class_id)}"
                for class_id in sv_detections.class_id  # type: ignore
            ]
            annotated_im = label_annotator.annotate(scene=annotated_im, detections=sv_detections, labels=labels)

        return Image.fromarray(annotated_im)
