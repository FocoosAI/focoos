# Copyright (c) FocoosAI
# Part of the code has been copied and adapted from Detectron2 (c) Facebook, Inc. and its affiliates.
import logging
import os
from typing import IO, Any, Dict, Iterable, List, Optional, Tuple, cast
from urllib.parse import parse_qs, urlparse

import torch
import torch.nn as nn
from iopath.common.file_io import HTTPURLHandler, PathManager
from torch.nn.parallel import DataParallel, DistributedDataParallel

from focoos.models.focoos_model import BaseModelNN
from focoos.utils.checkpoint import IncompatibleKeys
from focoos.utils.distributed import comm
from focoos.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["Checkpointer", "PeriodicCheckpointer"]


class Checkpointer:
    """
    A checkpointer that can save/load model as well as extra checkpointable
    objects.
    """

    def __init__(
        self,
        model: BaseModelNN,
        save_dir: str = "",
        *,
        save_to_disk: bool = True,
        **checkpointables: Any,
    ) -> None:
        """
        Args:
            model (nn.Module): model.
            save_dir (str): a directory to save and find checkpoints.
            save_to_disk (bool): if True, save checkpoint to disk, otherwise
                disable saving for this checkpointer.
            checkpointables (object): any checkpointable objects, i.e., objects
                that have the ``state_dict()`` and ``load_state_dict()`` method. For
                example, it can be used like
                `Checkpointer(model, "dir", optimizer=optimizer)`.
        """
        save_to_disk = comm.is_main_process() and save_to_disk
        if isinstance(model, (DistributedDataParallel, DataParallel)):
            model = model.module
        self.model = model
        self.checkpointables: Dict[str, Any] = {}
        for k, v in checkpointables.items():
            self.add_checkpointable(k, v)
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        # Default PathManager, support HTTP URLs (for backward compatibility in open source).
        # A user may want to use a different project-specific PathManager
        self.path_manager: PathManager = PathManager()
        self.path_manager.register_handler(HTTPURLHandler())
        self._parsed_url_during_load = None

    def add_checkpointable(self, key: str, checkpointable: Any) -> None:
        """
        Add checkpointable object for this checkpointer to track.

        Args:
            key (str): the key used to save the object
            checkpointable: any object with ``state_dict()`` and
                ``load_state_dict()`` method
        """
        if key in self.checkpointables:
            raise KeyError(f"Key {key} already used in the Checkpointer")
        if not hasattr(checkpointable, "state_dict"):
            raise TypeError("add_checkpointable needs an object with 'state_dict()' method.")
        self.checkpointables[key] = checkpointable

    def save(self, name: str, **kwargs: Any) -> None:
        """
        Dump model and checkpointables to a file.

        Args:
            name (str): name of the file.
            kwargs (dict): extra arbitrary data to save.
        """
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        with self.path_manager.open(save_file, "wb") as f:
            torch.save(data, cast(IO[bytes], f))
        self.tag_last_checkpoint(basename)

    def load(self, path: str, checkpointables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Load from the given checkpoint.

        Args:
            path (str): path or url to the checkpoint. If empty, will not load
                anything.
            checkpointables (list): List of checkpointable names to load. If not
                specified (None), will load all the possible checkpointables.
        Returns:
            dict:
                extra data loaded from the checkpoint that has not been
                processed. For example, those saved with
                :meth:`.save(**extra_data)`.
        """
        assert self._parsed_url_during_load is None
        need_sync = False
        logger.info("Loading from {} ...".format(path))

        if path and isinstance(self.model, DistributedDataParallel):
            has_file = os.path.isfile(path)
            all_has_file = comm.all_gather(has_file)
            if not all_has_file[0]:
                raise OSError(f"File {path} not found on main worker.")
            if not all(all_has_file):
                logger.warning(f"Not all workers can read checkpoint {path}. Training may fail to fully resume.")
                need_sync = True
            if not has_file:
                path = None  # type: ignore # don't load if not readable

        if not path:
            # no checkpoint provided
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        else:
            parsed_url = urlparse(path)
            self._parsed_url_during_load = parsed_url
            path = parsed_url._replace(query="").geturl()  # remove query from filename

        self.logger.info("[Checkpointer] Loading from {} ...".format(path))
        if not os.path.isfile(path):
            path = self.path_manager.get_local_path(path)
            assert os.path.isfile(path), "Checkpoint {} not found!".format(path)

        # Load and preprocess checkpoint file
        checkpoint = self._load_file(path)
        # Load the checkpoint into the model
        _ = self._load_model(checkpoint)

        for key in self.checkpointables if checkpointables is None else checkpointables:
            if key in checkpoint:
                self.logger.info("Loading {} from {} ...".format(key, path))
                obj = self.checkpointables[key]
                obj.load_state_dict(checkpoint.pop(key))

        # return any further checkpoint data
        ret = checkpoint
        if need_sync:
            logger.info("Broadcasting model states from main worker ...")
            self.model._sync_params_and_buffers()
        self._parsed_url_during_load = None  # reset to None
        return ret

    def has_checkpoint(self) -> bool:
        """
        Returns:
            bool: whether a checkpoint exists in the target directory.
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return self.path_manager.exists(save_file)

    def get_checkpoint_file(self) -> str:
        """
        Returns:
            str: The latest checkpoint file in target directory.
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with self.path_manager.open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            return ""
        return os.path.join(self.save_dir, last_saved)  # type: ignore

    def get_all_checkpoint_files(self) -> List[str]:
        """
        Returns:
            list: All available checkpoint files (.pth files) in target
                directory.
        """
        all_model_checkpoints = [
            os.path.join(self.save_dir, file)
            for file in self.path_manager.ls(self.save_dir)
            if self.path_manager.isfile(os.path.join(self.save_dir, file)) and file.endswith(".pth")
        ]
        return all_model_checkpoints

    def resume_or_load(self, path: str, *, resume: bool = True) -> Dict[str, Any]:
        """
        If `resume` is True, this method attempts to resume from the last
        checkpoint, if exists. Otherwise, load checkpoint from the given path.
        This is useful when restarting an interrupted training job.

        Args:
            path (str): path to the checkpoint.
            resume (bool): if True, resume from the last checkpoint if it exists
                and load the model together with all the checkpointables. Otherwise
                only load the model without loading any checkpointables.

        Returns:
            same as :meth:`load`.
        """
        if resume and self.has_checkpoint():
            path = self.get_checkpoint_file()
            return self.load(path)
        else:
            return self.load(path, checkpointables=[])

    def tag_last_checkpoint(self, last_filename_basename: str) -> None:
        """
        Tag the last checkpoint.

        Args:
            last_filename_basename (str): the basename of the last filename.
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with self.path_manager.open(save_file, "w") as f:
            f.write(last_filename_basename)  # type: ignore

    def _load_file(self, filename: str) -> Dict[str, Any]:
        """
        Load a checkpoint file. Can be overwritten by subclasses to support
        different formats.

        Args:
            filename (str): a locally mounted file path.
        Returns:
            dict: with keys "model" and optionally others that are saved by
                the checkpointer dict["model"] must be a dict which maps strings
                to torch.Tensor or numpy arrays.
        """
        loaded = torch.load(filename, map_location=torch.device("cpu"), weights_only=True)
        if "model" not in loaded:
            loaded = {"model": loaded}
        assert self._parsed_url_during_load is not None, "`_load_file` must be called inside `load`"
        parsed_url = self._parsed_url_during_load
        queries = parse_qs(parsed_url.query)

        if len(queries) > 0:  # probably can be removed, from detectron2
            logger.error(f"Unsupported query remaining: f{queries}, orginal filename: {parsed_url.geturl()}")
            raise ValueError(f"Unsupported query remaining: f{queries}, orginal filename: {parsed_url.geturl()}")
        return loaded

    def _load_model(self, checkpoint: Any) -> IncompatibleKeys:
        """
        Load weights from a checkpoint.

        Args:
            checkpoint (Any): checkpoint contains the weights.

        Returns:
            ``NamedTuple`` with ``missing_keys``, ``unexpected_keys``,
                and ``incorrect_shapes`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
                * **incorrect_shapes** is a list of (key, shape in checkpoint, shape in model)

            This is just like the return value of
            :func:`torch.nn.Module.load_state_dict`, but with extra support
            for ``incorrect_shapes``.
        """
        checkpoint_state_dict = checkpoint.pop("model")

        incompatible = self.model.load_state_dict(checkpoint_state_dict, strict=False)

        return incompatible


class PeriodicCheckpointer:
    """
    Save checkpoints periodically. When `.step(iteration)` is called, it will
    execute `checkpointer.save` on the given checkpointer, if iteration is a
    multiple of period or if `max_iter` is reached.

    Attributes:
        checkpointer (Checkpointer): the underlying checkpointer object
    """

    def __init__(
        self,
        checkpointer: Checkpointer,
        period: int,
        max_iter: Optional[int] = None,
        max_to_keep: Optional[int] = None,
        file_prefix: str = "model",
    ) -> None:
        """
        Args:
            checkpointer: the checkpointer object used to save checkpoints.
            period (int): the period to save checkpoint.
            max_iter (int): maximum number of iterations. When it is reached,
                a checkpoint named "{file_prefix}_final" will be saved.
            max_to_keep (int): maximum number of most current checkpoints to keep,
                previous checkpoints will be deleted
            file_prefix (str): the prefix of checkpoint's filename
        """
        self.checkpointer = checkpointer
        self.period = int(period)
        self.max_iter = max_iter
        if max_to_keep is not None:
            assert max_to_keep > 0
        self.max_to_keep = max_to_keep
        self.recent_checkpoints: List[str] = []
        self.path_manager: PathManager = checkpointer.path_manager
        self.file_prefix = file_prefix

    def step(self, iteration: int, **kwargs: Any) -> None:
        """
        Perform the appropriate action at the given iteration.

        Args:
            iteration (int): the current iteration, ranged in [0, max_iter-1].
            kwargs (Any): extra data to save, same as in
                :meth:`Checkpointer.save`.
        """
        iteration = int(iteration)
        additional_state = {"iteration": iteration}
        additional_state.update(kwargs)

        if (iteration + 1) % self.period == 0:
            self.checkpointer.save("{}_{:07d}".format(self.file_prefix, iteration), **additional_state)

            if self.max_to_keep is not None:
                self.recent_checkpoints.append(self.checkpointer.get_checkpoint_file())
                if len(self.recent_checkpoints) > self.max_to_keep:
                    file_to_delete = self.recent_checkpoints.pop(0)
                    if self.path_manager.exists(file_to_delete) and not file_to_delete.endswith(
                        f"{self.file_prefix}_final.pth"
                    ):
                        self.path_manager.rm(file_to_delete)

        if self.max_iter is not None:
            if iteration >= self.max_iter - 1:
                self.checkpointer.save(f"{self.file_prefix}_final", **additional_state)

    def save(self, name: str, **kwargs: Any) -> None:
        """
        Same argument as :meth:`Checkpointer.save`.
        Use this method to manually save checkpoints outside the schedule.

        Args:
            name (str): file name.
            kwargs (Any): extra data to save, same as in
                :meth:`Checkpointer.save`.
        """
        self.checkpointer.save(name, **kwargs)


def _named_modules_with_dup(model: nn.Module, prefix: str = "") -> Iterable[Tuple[str, nn.Module]]:
    """
    The same as `model.named_modules()`, except that it includes
    duplicated modules that have more than one name.
    """
    yield prefix, model
    for name, module in model._modules.items():
        if module is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        yield from _named_modules_with_dup(module, submodule_prefix)
