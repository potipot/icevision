__all__ = ["convert_raw_prediction"]

from icevision.imports import *
from icevision.core import *


def convert_raw_prediction(raw_pred: Sequence[np.ndarray], detection_threshold: float):
    stack_pred = np.vstack(raw_pred)

    scores = stack_pred[:, -1]
    bboxes = stack_pred[:, :-1]

    # each item in raw_pred is an array of predictions of it's `i` class
    labels = [np.full(o.shape[0], i, dtype=np.int32) for i, o in enumerate(raw_pred)]
    labels = np.concatenate(labels)

    keep_mask = scores > detection_threshold
    keep_scores = scores[keep_mask]
    keep_labels = labels[keep_mask]
    keep_bboxes = [BBox.from_xyxy(*o) for o in bboxes[keep_mask]]

    return {
        "scores": keep_scores,
        "labels": keep_labels,
        "bboxes": keep_bboxes,
    }
