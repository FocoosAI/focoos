import json
import os
from collections import defaultdict

from focoos.trainer.events import get_event_storage
from focoos.trainer.hooks.base import EventWriter
from focoos.utils.logger import get_logger
from focoos.utils.metrics import is_json_compatible

logger = get_logger(__name__)


class JSONWriter(EventWriter):
    """
    Write scalars to a json file as a single JSON array.

    Example structure of the resulting file:
    [
        {"iteration": 1, "loss": 1.0, ...},
        {"iteration": 2, "loss": 0.9, ...}
    ]
    """

    def __init__(self, json_file, window_size=20):
        """
        Args:
            json_file (str): path to the json file. New data will be inserted before the final "]".
                If the file doesn't exist, it will be created.
            window_size (int): the window size of median smoothing for the scalars whose
                `smoothing_hint` are True.
            force_close (bool): whether to close the file after each write operation.
        """
        self._json_file = json_file
        self._window_size = window_size
        self._last_write = -1

        # Initialize the file if it doesn't exist
        if not os.path.exists(json_file):
            with open(json_file, "w") as f:
                f.write("[\n]")

    def write(self):
        storage = get_event_storage()
        to_save = defaultdict(dict)

        # Get latest metrics with smoothing applied based on window_size
        for k, (v, iter) in storage.latest_with_smoothing_hint(self._window_size).items():
            if iter <= self._last_write:
                continue
            # Only save values that are JSON compatible
            if is_json_compatible(v):
                to_save[iter][k] = v

        # If we have new metrics to save
        if len(to_save):
            all_iters = sorted(to_save.keys())
            self._last_write = max(all_iters)

            # Open file in read+ mode
            with open(self._json_file, "r+") as f:
                # Read file contents
                f.seek(0, os.SEEK_END)
                file_size = f.tell()

                # Empty file or improperly formatted
                if file_size <= 2:
                    f.seek(0)
                    f.write("[\n]")
                    file_size = 3

                # Move cursor before the final bracket
                f.seek(file_size - 3)

                # Check if we need to add a comma (not empty array)
                last_char = f.read(1)
                needs_comma = last_char != "["

                # Go back to position before closing bracket
                f.seek(file_size - 2)

                # Write each metric object
                for itr, scalars_per_iter in to_save.items():
                    scalars_per_iter["iteration"] = itr
                    json_str = json.dumps(scalars_per_iter, sort_keys=True)

                    if needs_comma:
                        f.write(",\n" + json_str)
                    else:
                        f.write(json_str)
                        needs_comma = True

                # Write closing bracket
                f.write("\n]")

                # Ensure file is flushed to disk
                f.flush()
                try:
                    os.fsync(f.fileno())
                except AttributeError:
                    logger.warning("File handle doesn't support fsync")

    def close(self):
        pass
