from focoos.data.datasets.dict_dataset import DictDataset
from focoos.ports import Task
from focoos.trainer.evaluation.detection_evaluation import DetectionEvaluator, InstanceSegmentationEvaluator
from focoos.trainer.evaluation.sem_seg_evaluation import SemSegEvaluator

evaluators = {
    Task.DETECTION: DetectionEvaluator,
    Task.INSTANCE_SEGMENTATION: InstanceSegmentationEvaluator,
    Task.SEMSEG: SemSegEvaluator,
    # Task.PANOPTIC_SEGMENTATION: PanopticEvaluator,
}


def get_evaluator(dataset_dict: DictDataset, task: Task):
    if task in evaluators:
        return evaluators[task].from_datasetdict(dataset_dict=dataset_dict)
    else:
        raise ValueError(f"Task {task} not supported")
