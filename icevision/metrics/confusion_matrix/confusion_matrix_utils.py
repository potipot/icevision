from icevision.imports import *
from icevision import BBox, BaseRecord


@dataclass
class ObjectDetectionItem:
    bbox: BBox
    label: str
    label_id: int


@dataclass
class ObjectDetectionTarget(ObjectDetectionItem):
    iou_score: float = None
    matches: Collection[ObjectDetectionItem] = dataclasses.field(default_factory=list)


@dataclass
class ObjectDetectionPrediction(ObjectDetectionItem):
    score: float
    iou_score: float = None
    matches: Collection[ObjectDetectionItem] = dataclasses.field(default_factory=list)


def get_best_score_item(prediction_items: Collection[Dict]):
    # fill with dummy if list of prediction_items is empty
    dummy = ObjectDetectionPrediction(
        bbox=BBox.from_xyxy(0, 0, 0, 0),
        score=1.0,
        iou_score=1.0,
        label_id=0,
        label="background",
    )
    best_item = max(prediction_items, key=lambda x: x.score, default=dummy)
    return best_item


def pairwise_iou_record_record(target: BaseRecord, prediction: BaseRecord):
    """
    Calculates pairwise iou on prediction and target BaseRecord. Uses torchvision implementation of `box_iou`.
    """
    stacked_preds = [bbox.to_tensor() for bbox in prediction.detection.bboxes]
    stacked_preds = torch.stack(stacked_preds) if stacked_preds else torch.empty(0, 4)

    stacked_targets = [bbox.to_tensor() for bbox in target.detection.bboxes]
    stacked_targets = (
        torch.stack(stacked_targets) if stacked_targets else torch.empty(0, 4)
    )
    return torchvision.ops.box_iou(stacked_preds, stacked_targets)


def build_target_list(target: BaseRecord) -> List:
    target_list = [
        ObjectDetectionTarget(bbox=bbox, label=label, label_id=label_id)
        for bbox, label, label_id in zip(
            target.detection.bboxes, target.detection.labels, target.detection.label_ids
        )
    ]

    return target_list


def build_prediction_list(prediction: BaseRecord) -> List:
    prediction_list = [
        ObjectDetectionPrediction(
            bbox=bbox, label=label, label_id=label_id, score=score
        )
        for bbox, label, label_id, score in zip(
            prediction.detection.bboxes,
            prediction.detection.labels,
            prediction.detection.label_ids,
            prediction.detection.scores,
        )
    ]

    return prediction_list


def match_predictions_to_targets(
    target: BaseRecord, prediction: BaseRecord, iou_threshold: float = 0.5
) -> Collection:
    """
    matches bboxes, labels from targets with their predictions by iou threshold
    """
    # here we get a tensor of indices that match iou criteria (order is (pred_id, target_id))
    iou_table = pairwise_iou_record_record(target=target, prediction=prediction)
    pairs_indices = torch.nonzero(iou_table > iou_threshold)

    target_list = build_target_list(target)
    prediction_list = build_prediction_list(prediction)
    # creating a list of [target, matching_predictions]
    targets_with_predictions = [[target, []] for target in target_list]

    # appending matches to targets
    for pred_id, target_id in pairs_indices:
        single_prediction = deepcopy(prediction_list[pred_id])
        # python value casting needs rounding cause otherwise there are 0.69999991 values
        iou_score = round(iou_table[pred_id, target_id].item(), 4)
        single_prediction.iou_score = iou_score
        # seems like a magic number, but we want to append to the list of target's matching_predictions
        target_list[target_id].matches.append(single_prediction)
        # targets_with_predictions[target_id][1].append(single_prediction)

    return target_list


def match_targets_to_predictions(
    target: BaseRecord, prediction: BaseRecord, iou_threshold: float = 0.5
) -> Collection:
    """
    matches bboxes, labels from targets with their predictions by iou threshold
    """
    # TODO: sort prediction data by score, filter to keep only 100
    # here we get a tensor of indices that match iou criteria (order is (pred_id, target_id))
    iou_table = pairwise_iou_record_record(target=target, prediction=prediction)
    pairs_indices = torch.nonzero(iou_table > iou_threshold)

    target_list = build_target_list(target)
    prediction_list = build_prediction_list(prediction)
    # creating a list of [prediction, matching_targets]
    predictions_with_targets = [[prediction, []] for prediction in prediction_list]
    # appending matches to targets

    already_used = [False] * len(target.detection.bboxes)
    for pred_id, target_id in pairs_indices:
        single_target = target_list[target_id]
        # python value casting needs rounding cause otherwise there are 0.69999991 values
        iou_score = round(iou_table[pred_id, target_id].item(), 4)
        single_target.iou_score = iou_score
        # seems like a magic number, but we want to append to the list of target's matching_predictions
        if not already_used[target_id]:
            predictions_with_targets[pred_id][1].append(single_target)

    return predictions_with_targets


def filter_matched_targets_by_label_id(label_id, matched_targets):
    return [
        [pred, [target for target in targets if target["target_label_id"] == label_id]]
        for pred, targets in matched_targets
        if pred["predicted_label_id"] == label_id
    ]


class NoCopyRepeat(nn.Module):
    def __init__(self, out_channels=3):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x):
        return x.expand(self.out_channels, -1, -1)
