import os
from pathlib import Path
from typing import Tuple

from PIL import Image


def get_output_shape(old_height: int, old_width: int, short_edge_length: int, max_size: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target short edge length.
    """
    h, w = old_height, old_width
    size = short_edge_length * 1.0
    scale = size / min(h, w)
    if h < w:
        newh, neww = size, scale * w
    else:
        newh, neww = scale * h, size
    if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh = newh * scale
        neww = neww * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)

    return neww, newh


def resize_shortest_length(
    im_path: str,
    out_path: str,
    shortest_length: int = 1024,
    max_size: int = 2048,
    is_mask: bool = False,
):
    im_name = Path(im_path).name
    out_path = os.path.join(out_path, im_name)
    im = Image.open(im_path)
    new_width, new_height = get_output_shape(
        old_width=im.size[0],
        old_height=im.size[1],
        short_edge_length=shortest_length,
        max_size=max_size,
    )
    if is_mask:
        # mask = np.array(im,dtype=np.uint8)
        # mask = np.zeros((new_height, new_width), dtype=np.uint8)
        # print(mask.shape,mask.max())
        # im = Image.fromarray(mask.astype(np.uint8))
        im = im.resize((new_width, new_height), resample=Image.Resampling.NEAREST)
    else:
        im = im.resize((new_width, new_height))
    im.save(out_path)
