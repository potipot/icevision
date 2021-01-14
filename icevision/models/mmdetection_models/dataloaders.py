__all__ = [
    "build_train_batch",
    "build_valid_batch",
    "build_infer_batch",
    "train_dl",
    "valid_dl",
    "infer_dl",
]

from icevision.imports import *
from icevision.core import *
from icevision.models.utils import *


def train_dl(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    return transform_dl(
        dataset=dataset,
        build_batch=build_train_batch,
        batch_tfms=batch_tfms,
        **dataloader_kwargs
    )


def valid_dl(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    return transform_dl(
        dataset=dataset,
        build_batch=build_valid_batch,
        batch_tfms=batch_tfms,
        **dataloader_kwargs
    )


def infer_dl(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    """A `DataLoader` with a custom `collate_fn` that batches items as required for inferring the model.

    # Arguments
        dataset: Possibly a `Dataset` object, but more generally, any `Sequence` that returns records.
        batch_tfms: Transforms to be applied at the batch level.
        **dataloader_kwargs: Keyword arguments that will be internally passed to a Pytorch `DataLoader`.
        The parameter `collate_fn` is already defined internally and cannot be passed here.

    # Returns
        A Pytorch `DataLoader`.
    """
    return transform_dl(
        dataset=dataset,
        build_batch=build_infer_batch,
        batch_tfms=batch_tfms,
        **dataloader_kwargs
    )


def build_valid_batch(
    records: Sequence[RecordType], batch_tfms=None
) -> Tuple[dict, List[Dict[str, torch.Tensor]]]:
    return build_train_batch(records=records, batch_tfms=batch_tfms)


def build_train_batch(
    records: Sequence[RecordType], batch_tfms=None
) -> Tuple[dict, List[Dict[str, torch.Tensor]]]:
    records = common_build_batch(records=records, batch_tfms=batch_tfms)

    images, labels, bboxes, img_metas = [], [], [], []
    for record in records:
        image, target = _build_train_sample(record)
        images.append(image)
        labels.append(target["labels"])
        bboxes.append(target["boxes"])

        img_c, img_h, img_w = image.shape
        img_metas.append(
            {
                # height and width from sample is before padding
                "img_shape": (record["height"], record["width"], img_c),
                "pad_shape": (img_h, img_w, img_c),
                "scale_factor": np.ones(4),  # TODO: is scale factor correct?
            }
        )

    data = {
        "img": torch.stack(images),
        "img_metas": img_metas,
        "gt_labels": labels,
        "gt_bboxes": bboxes,
    }

    return data, records


# TODO Refactor train_dl (this is from torchvision faster_rcnn)
def _build_train_sample(
    record: RecordType,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    assert len(record["labels"]) == len(record["bboxes"])

    image = im2tensor(record["img"])
    target = {}

    # TODO: Is negative sampling supported on mmdetection?
    if len(record["labels"]) == 0:
        target["labels"] = torch.zeros(0, dtype=torch.int64)
        target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
    else:
        target["labels"] = tensor(record["labels"], dtype=torch.int64)
        xyxys = [bbox.xyxy for bbox in record["bboxes"]]
        target["boxes"] = tensor(xyxys, dtype=torch.float32)

    return image, target


def build_infer_batch(records, batch_tfms=None):
    records = common_build_batch(records, batch_tfms=batch_tfms)

    images, img_metas = [], []
    for record in records:
        image = im2tensor(record["img"])
        images.append(image)

        img_c, img_h, img_w = image.shape
        img_metas.append(
            {
                # height and width from sample is before padding
                "img_shape": (record["height"], record["width"], img_c),
                "pad_shape": (img_h, img_w, img_c),
                "scale_factor": np.ones(4),  # TODO: is scale factor correct?
            }
        )

    data = {
        "img": [torch.stack(images)],
        "img_metas": [img_metas],
    }

    return data, records
