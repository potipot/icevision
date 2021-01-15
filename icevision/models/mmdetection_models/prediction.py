__all__ = ["bbox_convert_raw_prediction", "mask_convert_raw_prediction"]

from icevision.imports import *
from icevision.core import *


def _unpack_raw_bboxes(raw_bboxes):
    stack_raw_bboxes = np.vstack(raw_bboxes)

    scores = stack_raw_bboxes[:, -1]
    bboxes = stack_raw_bboxes[:, :-1]

    # each item in raw_pred is an array of predictions of it's `i` class
    labels = [np.full(o.shape[0], i, dtype=np.int32) for i, o in enumerate(raw_bboxes)]
    labels = np.concatenate(labels)

    return scores, labels, bboxes


def bbox_convert_raw_prediction(
    raw_pred: Sequence[np.ndarray], detection_threshold: float
):
    scores, labels, bboxes = _unpack_raw_bboxes

    keep_mask = scores > detection_threshold
    keep_scores = scores[keep_mask]
    keep_labels = labels[keep_mask]
    keep_bboxes = [BBox.from_xyxy(*o) for o in bboxes[keep_mask]]

    return {
        "scores": keep_scores,
        "labels": keep_labels,
        "bboxes": keep_bboxes,
    }


def mask_convert_raw_prediction(
    raw_pred: Sequence[np.ndarray], detection_threshold: float
):
    raw_bboxes, raw_masks = raw_pred
    scores, labels, bboxes = _unpack_raw_bboxes(raw_bboxes)

    keep_mask = scores > detection_threshold
    keep_scores = scores[keep_mask]
    keep_labels = labels[keep_mask]
    keep_bboxes = [BBox.from_xyxy(*o) for o in bboxes[keep_mask]]
    keep_masks = MaskArray(np.vstack(raw_masks)[keep_mask])

    return {
        "scores": keep_scores,
        "labels": keep_labels,
        "bboxes": keep_bboxes,
        "masks": keep_masks,
    }
