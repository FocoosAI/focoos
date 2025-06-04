from collections import OrderedDict
from typing import List

import torch

from focoos.data.datasets.dict_dataset import DictDataset
from focoos.data.mappers.classification_dataset_mapper import ClassificationDatasetDict
from focoos.models.fai_cls.ports import ClassificationModelOutput
from focoos.trainer.evaluation.evaluator import DatasetEvaluator
from focoos.utils.distributed.comm import all_gather, is_main_process, synchronize
from focoos.utils.logger import get_logger

logger = get_logger(__name__)


class ClassificationEvaluator(DatasetEvaluator):
    """Evaluator for classification tasks with comprehensive metrics computation.

    This evaluator computes various classification metrics including accuracy, precision,
    recall, and F1 score both per-class and as macro/weighted averages. It supports
    distributed evaluation across multiple processes.

    Attributes:
        dataset_dict (DictDataset): Dataset containing ground truth annotations.
        metadata: Metadata from the dataset containing class information.
        num_classes (int): Number of classes in the classification task.
        class_names (List[str]): Names of the classes.
    """

    def __init__(
        self,
        dataset_dict: DictDataset,
        distributed=True,
    ):
        """Initialize the ClassificationEvaluator.

        Args:
            dataset_dict (DictDataset): Dataset in DictDataset format containing
                the ground truth annotations.
            distributed (bool, optional): If True, evaluation will be distributed
                across multiple processes. Defaults to True.
        """
        self.dataset_dict = dataset_dict
        self.metadata = self.dataset_dict.metadata
        self._distributed = distributed
        self._cpu_device = torch.device("cpu")
        self.num_classes = self.metadata.num_classes
        self.class_names = self.metadata.thing_classes

        self._predictions = []
        self._targets = []

    @classmethod
    def from_datasetdict(cls, dataset_dict, **kwargs):
        """Create ClassificationEvaluator instance from a dataset dictionary.

        Args:
            dataset_dict: Dataset dictionary containing the data and metadata.
            **kwargs: Additional keyword arguments passed to the constructor.

        Returns:
            ClassificationEvaluator: New instance of the evaluator.
        """
        return cls(dataset_dict=dataset_dict, **kwargs)

    def reset(self):
        """Clear stored predictions and targets.

        This method resets the internal state of the evaluator by clearing
        all accumulated predictions and ground truth targets.
        """
        self._predictions = []
        self._targets = []

    def process(self, inputs: List[ClassificationDatasetDict], outputs: List[ClassificationModelOutput]):
        """Process a batch of inputs and outputs for evaluation.

        This method extracts predictions and ground truth labels from the provided
        inputs and outputs, then stores them for later evaluation.

        Args:
            inputs (List[ClassificationDatasetDict]): List of input dictionaries,
                each containing ground truth information. Expected to have 'label'
                field or 'annotations' with category_id.
            outputs (List[ClassificationModelOutput]): List of model outputs,
                each containing 'logits' field with predicted class logits or
                ClassificationModelOutput instances.

        Note:
            - Ground truth labels are extracted from input['label'] or
              input['annotations'][0]['category_id']
            - Predictions are extracted from output['logits'] or output.logits
            - Items with missing labels or logits are skipped with warnings
        """
        for input_item, output_item in zip(inputs, outputs):
            # Get ground truth label from input
            label = None
            if "label" in input_item:
                label = input_item["label"]
            elif hasattr(input_item, "label"):
                label = input_item.label
            elif "annotations" in input_item and len(input_item["annotations"]) > 0:
                # Handle label from annotations format
                label = input_item["annotations"][0].get("category_id", None)

            if label is None:
                logger.warning(f"Could not find label in input item: {input_item}")
                continue

            # Get model predictions from output
            logits = None
            if isinstance(output_item, dict) and "logits" in output_item:
                logits = output_item["logits"]
            elif hasattr(output_item, "logits"):
                # Handle ClassificationModelOutput objects
                logits = output_item.logits

            if logits is None:
                logger.warning(f"Could not find logits in output item: {output_item}")
                continue

            # Move tensors to CPU for evaluation
            logits = logits.to(self._cpu_device)

            # For image classification, logits will be [num_classes]
            # For batch processing, it could be [batch_size, num_classes]
            if logits.dim() > 1:
                # Assume first dimension is batch size
                predicted_class = torch.argmax(logits, dim=1).item()
            else:
                predicted_class = torch.argmax(logits, dim=0).item()

            # Store prediction and ground truth
            self._predictions.append(predicted_class)
            self._targets.append(label)

    def _compute_confusion_matrix(self, y_true, y_pred, num_classes):
        """Compute confusion matrix for classification evaluation.

        Args:
            y_true (torch.Tensor): Ground truth labels tensor.
            y_pred (torch.Tensor): Predicted labels tensor.
            num_classes (int): Number of classes in the classification task.

        Returns:
            torch.Tensor: Confusion matrix of shape (num_classes, num_classes).
                Element [i,j] represents the number of samples with true label i
                that were predicted as label j.
        """
        confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
        for t, p in zip(y_true, y_pred):
            confusion_matrix[t, p] += 1
        return confusion_matrix

    def evaluate(self):
        """Evaluate classification metrics on accumulated predictions.

        Computes comprehensive classification metrics including accuracy,
        per-class precision/recall/F1, and macro/weighted averages.

        Returns:
            OrderedDict: Dictionary containing evaluation metrics with the following keys:
                - 'Accuracy': Overall classification accuracy (%)
                - 'Macro-Precision': Macro-averaged precision (%)
                - 'Macro-Recall': Macro-averaged recall (%)
                - 'Macro-F1': Macro-averaged F1 score (%)
                - 'Weighted-Precision': Weighted precision (%)
                - 'Weighted-Recall': Weighted recall (%)
                - 'Weighted-F1': Weighted F1 score (%)
                - 'Precision-{class_name}': Per-class precision (%)
                - 'Recall-{class_name}': Per-class recall (%)
                - 'F1-{class_name}': Per-class F1 score (%)

        Note:
            - In distributed mode, only the main process returns results
            - All metrics are expressed as percentages
            - Returns empty dict if no predictions are available
            - Results are wrapped in 'classification' key for trainer compatibility
        """
        if self._distributed:
            synchronize()
            predictions_list = all_gather(self._predictions)
            targets_list = all_gather(self._targets)

            # Flatten gathered lists
            predictions = []
            targets = []
            for p_list, t_list in zip(predictions_list, targets_list):
                if p_list is not None:
                    predictions.extend(p_list)
                if t_list is not None:
                    targets.extend(t_list)

            if not is_main_process():
                return
        else:
            predictions = self._predictions
            targets = self._targets

        # Check if we have predictions to evaluate
        if len(predictions) == 0:
            logger.warning("No predictions to evaluate")
            return OrderedDict({"classification": {}})

        # Convert lists to tensors
        y_true = torch.tensor(targets, dtype=torch.long)
        y_pred = torch.tensor(predictions, dtype=torch.long)

        # Compute confusion matrix
        cm = self._compute_confusion_matrix(y_true, y_pred, self.num_classes)

        # Calculate accuracy
        accuracy = 100.0 * cm.diag().sum().float() / cm.sum().float()

        # Calculate per-class metrics
        tp = cm.diag()  # True positives for each class
        pred_sum = cm.sum(dim=0)  # Sum over actual classes (columns)
        target_sum = cm.sum(dim=1)  # Sum over predicted classes (rows)

        # Precision for each class
        precision_per_class = torch.zeros(self.num_classes, dtype=torch.float)
        for i in range(self.num_classes):
            if pred_sum[i] > 0:
                precision_per_class[i] = 100.0 * tp[i].float() / pred_sum[i].float()

        # Recall for each class
        recall_per_class = torch.zeros(self.num_classes, dtype=torch.float)
        for i in range(self.num_classes):
            if target_sum[i] > 0:
                recall_per_class[i] = 100.0 * tp[i].float() / target_sum[i].float()

        # F1 score for each class
        f1_per_class = torch.zeros(self.num_classes, dtype=torch.float)
        for i in range(self.num_classes):
            if precision_per_class[i] + recall_per_class[i] > 0:
                f1_per_class[i] = (
                    2 * (precision_per_class[i] * recall_per_class[i]) / (precision_per_class[i] + recall_per_class[i])
                )

        # Calculate macro averages
        valid_precision_classes = (pred_sum > 0).sum().item()
        macro_precision = precision_per_class.sum() / max(valid_precision_classes, 1)

        valid_recall_classes = (target_sum > 0).sum().item()
        macro_recall = recall_per_class.sum() / max(valid_recall_classes, 1)

        if macro_precision + macro_recall > 0:
            macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
        else:
            macro_f1 = torch.tensor(0.0)

        # Calculate weighted averages
        weights = target_sum.float() / target_sum.sum().float()
        weighted_precision = (precision_per_class * weights).sum()
        weighted_recall = (recall_per_class * weights).sum()

        if weighted_precision + weighted_recall > 0:
            weighted_f1 = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
        else:
            weighted_f1 = torch.tensor(0.0)

        # Create results dictionary
        results = OrderedDict()
        results["Accuracy"] = accuracy.item()
        results["Macro-Precision"] = macro_precision.item()
        results["Macro-Recall"] = macro_recall.item()
        results["Macro-F1"] = macro_f1.item()
        results["Weighted-Precision"] = weighted_precision.item()
        results["Weighted-Recall"] = weighted_recall.item()
        results["Weighted-F1"] = weighted_f1.item()

        # Add per-class metrics
        if self.class_names is not None:
            for i, class_name in enumerate(self.class_names):
                if i < self.num_classes:
                    results[f"Precision-{class_name}"] = precision_per_class[i].item()
                    results[f"Recall-{class_name}"] = recall_per_class[i].item()
                    results[f"F1-{class_name}"] = f1_per_class[i].item()

        # Log results
        logger.info("Classification Evaluation Results:")
        for k, v in results.items():
            logger.info(f"  {k}: {v:.2f}")

        # Return results in the expected format for trainer
        return OrderedDict({"classification": results})
