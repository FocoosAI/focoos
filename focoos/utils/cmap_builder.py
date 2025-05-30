from typing import Optional

import numpy as np


def cmap_builder(classes: Optional[list] = None, normalized: bool = False) -> np.ndarray:
    classes = list(range(256)) if classes is None else classes

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((256, 3), dtype=dtype) + 160
    for idx in classes:
        r = g = b = 0
        c = idx
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[idx] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap
