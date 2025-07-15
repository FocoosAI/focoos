import datetime
import time
from typing import Optional

import torch

from focoos.trainer.events import get_event_storage
from focoos.trainer.hooks.base import EventWriter
from focoos.utils.logger import get_logger

logger = get_logger("metrics")


class CommonMetricPrinter(EventWriter):
    """
    Print **common** metrics to the terminal, including
    iteration time, ETA, memory, all losses, and the learning rate.
    It also applies smoothing using a window of 20 elements.

    It's meant to print common metrics in common ways.
    To print something in more customized ways, please implement a similar printer by yourself.
    """

    def __init__(self, max_iter: Optional[int] = None, window_size: int = 20):
        """
        Args:
            max_iter: the maximum number of iterations to train.
                Used to compute ETA. If not given, ETA will not be printed.
            window_size (int): the losses will be median-smoothed by this window size
        """

        self._max_iter = max_iter
        self._window_size = window_size
        self._last_write = None  # (step, time) of last call to write(). Used to compute ETA

    def _get_eta(self, storage) -> Optional[str]:
        if self._max_iter is None:
            return ""
        iteration = storage.iter
        try:
            eta_seconds = storage.history("time").median(1000) * (self._max_iter - iteration - 1)
            storage.put_scalar("eta_seconds", eta_seconds, smoothing_hint=False)
            return str(datetime.timedelta(seconds=int(eta_seconds)))
        except KeyError:
            # estimate eta on our own - more noisy
            eta_string = None
            if self._last_write is not None:
                estimate_iter_time = (time.perf_counter() - self._last_write[1]) / (iteration - self._last_write[0])
                eta_seconds = estimate_iter_time * (self._max_iter - iteration - 1)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            self._last_write = (iteration, time.perf_counter())
            return eta_string

    def write(self):
        storage = get_event_storage()
        iteration = storage.iter
        if iteration == self._max_iter:
            # This hook only reports training progress (loss, ETA, etc) but not other data,
            # therefore do not write anything after training succeeds, even if this method
            # is called.
            return

        try:
            avg_data_time = storage.history("data_time").avg(storage.count_samples("data_time", self._window_size))
            last_data_time = storage.history("data_time").latest()
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            avg_data_time = None
            last_data_time = None
        try:
            avg_iter_time = storage.history("time").global_avg()
            last_iter_time = storage.history("time").latest()
        except KeyError:
            avg_iter_time = None
            last_iter_time = None
        try:
            lr = "{:.5g}".format(storage.history("lr").latest())
        except KeyError:
            lr = "N/A"

        eta_string = self._get_eta(storage)

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        logger.info(
            str.format(
                " {eta}iter: {iter}  {losses}  {non_losses}  {avg_time}{last_time}"
                + "{avg_data_time}{last_data_time} lr: {lr}  {memory}",
                eta=f"eta: {eta_string}  " if eta_string else "",
                iter=iteration,
                losses="  ".join(
                    [
                        "{}: {:.4g}".format(k, v.median(storage.count_samples(k, self._window_size)))
                        for k, v in storage.histories().items()
                        if "loss" in k
                    ]
                ),
                non_losses="  ".join(
                    [
                        "{}: {:.4g}".format(k, v.median(storage.count_samples(k, self._window_size)))
                        for k, v in storage.histories().items()
                        if "[metric]" in k
                    ]
                ),
                avg_time=("time: {:.4f}  ".format(avg_iter_time) if avg_iter_time is not None else ""),
                last_time=("last_time: {:.4f}  ".format(last_iter_time) if last_iter_time is not None else ""),
                avg_data_time=("data_time: {:.4f}  ".format(avg_data_time) if avg_data_time is not None else ""),
                last_data_time=(
                    "last_data_time: {:.4f}  ".format(last_data_time) if last_data_time is not None else ""
                ),
                lr=lr,
                memory=("max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else ""),
            )
        )
