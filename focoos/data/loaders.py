import operator
from typing import Union

import torch
import torch.utils.data as torchdata

from focoos.utils.distributed import comm
from focoos.utils.env import seed_all_rng
from focoos.utils.logger import get_logger

from .datasets.common import AspectRatioGroupedDataset, ToIterableDataset
from .datasets.map_dataset import MapDataset
from .samplers import InferenceSampler, TrainingSampler


def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2**31
    seed_all_rng(initial_seed + worker_id)


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def build_batch_data_loader(
    dataset,
    sampler,
    total_batch_size,
    *,
    aspect_ratio_grouping=False,
    num_workers=0,
    prefetch_factor=2, # DEBUG 2
    collate_fn=None,
    drop_last: bool = True,
    **kwargs,
) -> Union[torchdata.DataLoader, AspectRatioGroupedDataset]:
    """
    Build a batched dataloader. The main differences from `torch.utils.data.DataLoader` are:
    1. support aspect ratio grouping options
    2. use no "batch collation", because this is common for detection training

    Args:
        dataset (torch.utils.data.Dataset): a pytorch map-style or iterable dataset.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces indices.
            Must be provided iff. ``dataset`` is a map-style dataset.
        total_batch_size, aspect_ratio_grouping, num_workers, collate_fn: see
            :func:`build_detection_train_loader`.
        single_gpu_batch_size: You can specify either `single_gpu_batch_size` or `total_batch_size`.
            `single_gpu_batch_size` specifies the batch size that will be used for each gpu/process.
            `total_batch_size` allows you to specify the total aggregate batch size across gpus.
            It is an error to supply a value for both.
        drop_last (bool): if ``True``, the dataloader will drop incomplete batches.

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = comm.get_world_size()
    assert total_batch_size > 0 and total_batch_size % world_size == 0, (
        "Total batch size ({}) must be divisible by the number of gpus ({}).".format(total_batch_size, world_size)
    )
    batch_size = total_batch_size // world_size
    logger = get_logger(__name__)
    logger.info("Making batched data loader with batch_size=%d", batch_size)

    dataset = ToIterableDataset(dataset, sampler, shard_chunk_size=batch_size)

    if aspect_ratio_grouping:
        assert drop_last, "Aspect ratio grouping will drop incomplete batches."
        data_loader = torchdata.DataLoader(
            dataset,
            num_workers=num_workers,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
            prefetch_factor=prefetch_factor,
        )  # yield individual mapped dict
        data_loader = AspectRatioGroupedDataset(data_loader, batch_size)
        return data_loader
    else:
        return torchdata.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=num_workers,
            collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
            worker_init_fn=worker_init_reset_seed,
            prefetch_factor=prefetch_factor,
        )


def build_detection_train_loader(
    dataset: MapDataset,
    *,
    total_batch_size,
    aspect_ratio_grouping=True,
    num_workers=0,
    collate_fn=None,
) -> Union[torchdata.DataLoader, AspectRatioGroupedDataset]:
    """
    Build a dataloader for object detection with some default features.

    Args:
        dataset (MapDataset): a MapDataset,
        total_batch_size (int): total batch size across all workers.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers
        collate_fn: a function that determines how to do batching, same as the argument of
            `torch.utils.data.DataLoader`. Defaults to do no collation and return a list of
            data. No collation is OK for small batch size and simple data structures.
            If your batch size is large and each sample contains too many small tensors,
            it's more efficient to collate them in data loader.

    Returns:
        torch.utils.data.DataLoader:
            a dataloader. Each output from it is a ``list[mapped_element]`` of length
            ``total_batch_size / num_workers``, where ``mapped_element`` is produced
            by the ``mapper``.
    """

    return build_batch_data_loader(
        dataset=dataset,
        sampler=TrainingSampler(len(dataset)),
        total_batch_size=total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def build_detection_test_loader(
    dataset: MapDataset,
    *,
    batch_size: int = 1,
    num_workers: int = 0,
    collate_fn=None,
) -> torchdata.DataLoader:
    """
    Similar to `build_detection_train_loader`, with default batch size = 1,
    and sampler = :class:`InferenceSampler`. This sampler coordinates all workers
    to produce the exact set of all samples.

    Args:
        dataset (MapDataset): a MapDataset.
        batch_size (int): the batch size of the data loader to be created.
            Default to 1 image per worker since this is the standard when reporting
            inference time in papers.
        num_workers (int): number of parallel data loading workers
        collate_fn: same as the argument of `torch.utils.data.DataLoader`.
            Defaults to do no collation and return a list of data.

    Returns:
        torch.utils.data.DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(dataset, batch_size=1, num_workers=4)

        # or, with custom collate function:
        data_loader = build_detection_test_loader(dataset, batch_size=2, num_workers=2, collate_fn=my_custom_collate_fn)
    """

    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        sampler=InferenceSampler(len(dataset)),
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
    )
