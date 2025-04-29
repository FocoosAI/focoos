import random
from contextlib import ExitStack

import torch

from focoos.data.datasets.map_dataset import MapDataset
from focoos.models.fai_model import BaseModelNN
from focoos.trainer.events import get_event_storage
from focoos.utils.visualizer import ColorMode, Visualizer

from .base import HookBase


class VisualizationHook(HookBase):
    def __init__(
        self,
        model: BaseModelNN,
        dataset: MapDataset,
        period,
        index_list=[],
        n_sample=4,
        random_samples=False,
    ):
        self.model = model
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

    def _visualize(self):
        training_mode = self.model.training

        with ExitStack() as stack:
            stack.enter_context(torch.no_grad())

            storage = get_event_storage()
            self.model.eval()

            for i in range(self.n_sample):
                sample = self.samples[i]
                sample["height"], sample["width"] = sample["image"].shape[-2:]

                samples = [sample]
                prediction = self.model.post_process(self.model(samples), samples)[0]

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
                    vis_img = pred_img.transpose(2, 0, 1)
                    storage.put_image(f"Image_{i}", vis_img)

        self.model.train(training_mode)

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0:
            # do the last eval in after_train
            if next_iter != self.trainer.max_iter:
                self._visualize()

    def after_train(self):
        # This condition is to prevent the eval from running after a failed training
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self._visualize()
