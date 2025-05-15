import pickle

import numpy as np
import torch

from focoos.utils.logger import get_logger

logger = get_logger(__name__)


class TorchSerializedDataset:
    def __init__(self, lst: list):
        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        logger.debug("Serializing {} elements to byte tensors and concatenating them all ...".format(len(lst)))
        _lst = [_serialize(x) for x in lst]
        self._addr = np.asarray([len(x) for x in _lst], dtype=np.int64)
        self._addr = torch.from_numpy(np.cumsum(self._addr))
        self._lst = torch.from_numpy(np.concatenate(_lst))
        logger.debug("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024**2))

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr].numpy())
        return pickle.loads(bytes)
