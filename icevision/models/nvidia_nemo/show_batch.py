__all__ = ["show_batch"]

from icevision.utils import *
from icevision.visualize import *


def show_batch(
    batch, ncols: int = 1, figsize=None, denormalize_fn=None, **show_samples_kwargs
):
    """Show a single batch from a dataloader.
    # Arguments
        show_samples_kwargs: Check the parameters from `show_samples`
    """
    (tensor_images, *_), records = batch

    for tensor_image, record in zip(tensor_images, records):
        # nemo only works on single channel, squeezed data
        assert tensor_image.ndim == 1
        record.set_wav(tensor_image.unsqueeze(0))

    return show_samples(
        records,
        ncols=ncols,
        figsize=figsize,
        denormalize_fn=denormalize_fn,
        **show_samples_kwargs
    )
