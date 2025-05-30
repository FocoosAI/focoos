import torch

from focoos.utils.logger import get_logger

logger = get_logger(__name__)


def network_to_half(model):
    """
    Convert model to half precision in a batchnorm-safe way.
    """
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    def bn_to_float(module):
        """
        BatchNorm layers need parameters in single precision. Find all layers and convert
        them back to float.
        """
        if isinstance(module, norm_module_types):
            module.float()
        for child in module.children():
            bn_to_float(child)
        return module

    return bn_to_float(model.half())


def torch_export(
    model_name,
    model,
    size,
    device="cuda",
    weights=None,
    fp16=False,
    dtype=torch.uint8,
):
    torch.cuda.empty_cache()
    model.to(device)
    model.eval()

    if weights is not None:
        ckpt = torch.load(weights, map_location=torch.device("cpu"))["model"]
        model.load_state_dict(ckpt, strict=True)
        logger.info(f"Weights loaded from {weights}")

    if not (isinstance(size, tuple) or isinstance(size, list)):
        size = [size, size]
    else:
        assert len(size) == 2

    data = 128 * torch.ones(1, 3, size[0], size[1], dtype=dtype).to(device)
    with torch.no_grad():
        if fp16:
            model = network_to_half(model)

        model = torch.jit.trace(model, data)
        logger.info("Correctly traced model.")
        logger.debug(model.graph)

    model.save(model_name)
    logger.info(f"Correctly saved model at {model_name}.")

    return model
