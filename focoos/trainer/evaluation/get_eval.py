from focoos.data.datasets.dict_dataset import DictDataset
from focoos.ports import Task


def get_evaluator(dataset_dict: DictDataset, task: Task):
    if task == Task.DETECTION:
        from focoos.trainer.evaluation.detection_evaluation import DetectionEvaluator

        return DetectionEvaluator.from_datasetdict(dataset_dict=dataset_dict)
    elif task == Task.INSTANCE_SEGMENTATION:
        from focoos.trainer.evaluation.detection_evaluation import InstanceSegmentationEvaluator

        return InstanceSegmentationEvaluator.from_datasetdict(dataset_dict=dataset_dict)
    elif task == Task.SEMSEG:
        from focoos.trainer.evaluation.sem_seg_evaluation import SemSegEvaluator

        return SemSegEvaluator.from_datasetdict(dataset_dict=dataset_dict)
    elif task == Task.CLASSIFICATION:
        from focoos.trainer.evaluation.classification_evaluation import ClassificationEvaluator

        return ClassificationEvaluator.from_datasetdict(dataset_dict=dataset_dict)
    # elif task == Task.PANOPTIC_SEGMENTATION:
    #     from focoos.trainer.evaluation.panoptic_evaluation import PanopticEvaluator
    #     return PanopticEvaluator.from_datasetdict(dataset_dict=dataset_dict)
    else:
        raise ValueError(f"Task {task} not supported")
