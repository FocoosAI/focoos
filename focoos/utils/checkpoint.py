from collections import defaultdict
from typing import Any, Dict, List, Tuple

from termcolor import colored

from focoos.utils.logger import get_logger

logger = get_logger("Checkpoint")


def strip_prefix_if_present(state_dict: Dict[str, Any], prefix: str) -> None:
    """
    Strip the prefix in metadata, if any.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return

    for key in keys:
        newkey = key[len(prefix) :]
        state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata, if any..
    try:
        metadata = state_dict._metadata  # type: ignore
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)


def _group_checkpoint_keys(keys: List[str]) -> Dict[str, List[str]]:
    """
    Group keys based on common prefixes. A prefix is the string up to the final
    "." in each key.
    Args:
        keys (list[str]): list of parameter names, i.e. keys in the model
            checkpoint dict.
    Returns:
        dict[list]: keys with common prefixes are grouped into lists.
    """
    groups = defaultdict(list)
    for key in keys:
        pos = key.rfind(".")
        if pos >= 0:
            head, tail = key[:pos], [key[pos + 1 :]]
        else:
            head, tail = key, []
        groups[head].extend(tail)
    return groups


def _group_to_str(group: List[str]) -> str:
    """
    Format a group of parameter name suffixes into a loggable string.
    Args:
        group (list[str]): list of parameter name suffixes.
    Returns:
        str: formated string.
    """
    if len(group) == 0:
        return ""

    if len(group) == 1:
        return "." + group[0]

    return ".{" + ", ".join(sorted(group)) + "}"


def get_missing_parameters_message(keys: List[str]) -> str:
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the model but not found in a checkpoint.
    Args:
        keys (list[str]): List of keys that were not found in the checkpoint.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg_per_group = sorted(k + _group_to_str(v) for k, v in groups.items())
    msg = "Some model parameters or buffers are not found in the checkpoint:\n"
    msg += "\n".join([colored(x, "blue") for x in msg_per_group])
    return msg


def get_unexpected_parameters_message(keys: List[str]) -> str:
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the checkpoint but not found in the model.
    Args:
        keys (list[str]): List of keys that were not found in the model.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = "The checkpoint state_dict contains keys that are not used by the model:\n"
    msg += "\n".join("  " + colored(k + _group_to_str(v), "magenta") for k, v in groups.items())
    return msg


class IncompatibleKeys:
    def __init__(
        self,
        missing_keys: List[str],
        unexpected_keys: List[str],
        incorrect_shapes: List[Tuple[str, Tuple[int], Tuple[int]]],
    ):
        self.missing_keys = missing_keys
        self.unexpected_keys = unexpected_keys
        self.incorrect_shapes = incorrect_shapes

    def __str__(self):
        return f"Missing keys: {self.missing_keys}\nUnexpected keys: {self.unexpected_keys}\nIncorrect shapes: {self.incorrect_shapes}"

    def __repr__(self):
        return self.__str__()

    def log_incompatible_keys(self) -> None:
        """
        Log information about the incompatible keys returned by ``_load_model``.
        """
        for k, shape_checkpoint, shape_model in self.incorrect_shapes:
            logger.warning(
                "Skip loading parameter '{}' to the model due to incompatible "
                "shapes: {} in the checkpoint but {} in the "
                "model! You might want to double check if this is expected.".format(k, shape_checkpoint, shape_model)
            )
        if self.missing_keys:
            missing_keys = self.missing_keys
            if missing_keys:
                logger.warning(get_missing_parameters_message(missing_keys))
        if self.unexpected_keys:
            logger.warning(get_unexpected_parameters_message(self.unexpected_keys))
