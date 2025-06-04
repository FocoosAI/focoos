# Copyright (c) Facebook, Inc. and its affiliates.
from collections.abc import Mapping

from focoos.utils.logger import get_logger

logger = get_logger(__name__)


def print_csv_format(results):
    """
    Print main metrics in a format similar to Detectron,
    so that they are easy to copypaste into a spreadsheet.

    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
            unordered dict can also be printed, but in arbitrary order
    """
    assert isinstance(results, Mapping) or not len(results), results
    for task, res in results.items():
        if isinstance(res, Mapping):
            # Don't print "AP-category" metrics since they are usually not tracked.
            important_res = [(k, v) for k, v in res.items() if "-" not in k]
            logger.info("copypaste: Task: {}".format(task))
            logger.info("copypaste: " + ",".join([k[0] for k in important_res]))
            logger.info("copypaste: " + ",".join([f"{(k[1] if k[1] is not None else -1):.4f}" for k in important_res]))
        else:
            logger.info(f"copypaste: {task}={res}")
