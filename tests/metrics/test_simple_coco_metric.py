from icevision.all import *
from review_object_detection_metrics.src.evaluators.coco_evaluator import (
    get_coco_summary,
)
from review_object_detection_metrics.src.bounding_box import BoundingBox


def prediction_to_rodm_lists(predictions: List[Prediction]) -> Tuple[List, List]:
    detected_bbs = []
    groundtruth_bbs = []
    for prediction in predictions:
        d = prediction.pred.detection
        for id, bbox, label, label_id, score in zip(
            [d.record_id] * len(d.bboxes), d.bboxes, d.labels, d.label_ids, d.scores
        ):
            detected_bbs.append(
                BoundingBox(
                    image_name=id,
                    class_id=label_id,
                    coordinates=bbox.xywh,
                    confidence=score,
                )
            )
        gt = prediction.ground_truth.detection
        for id, bbox, label, label_id in zip(
            [gt.record_id] * len(gt.bboxes), gt.bboxes, gt.labels, gt.label_ids
        ):
            groundtruth_bbs.append(
                BoundingBox(
                    image_name=id,
                    class_id=label_id,
                    coordinates=bbox.xywh,
                )
            )
    return groundtruth_bbs, detected_bbs


class IdmapRecordComponent(RecordComponent):
    def set_idmap(self, idmap: IDMap):
        self.idmap = idmap


class COCODummyParser(parsers.COCOBBoxParser):
    def img_size(self, o):
        return ImgSize(0, 0)

    def parse_fields(self, o, record, is_new):
        super().parse_fields(o, record, is_new)
        record.set_idmap(self.idmap)

    def template_record(self):
        record = super().template_record()
        record.add_component(IdmapRecordComponent())
        return record


class COCODetectionsParser(parsers.Parser):
    def img_size(self, o):
        return ImgSize(1, 1)

    def __init__(
        self,
        annotations_filepath: Union[str, Path],
        img_dir: Union[str, Path],
        class_map: ClassMap,
        idmap: Optional[IDMap] = None,
    ):

        self.annotations = json.loads(Path(annotations_filepath).read_bytes())
        self.img_dir = Path(img_dir)
        self.class_map = class_map
        super().__init__(self.template_record(), None)

    def prepare(self, o):
        pass

    def filepath(self, o) -> Path:
        return Path()

    def __iter__(self):
        yield from self.annotations

    def parse_fields(self, o, record, is_new):
        record.set_idmap(self.idmap)
        record.detection.set_class_map(self.class_map)
        record.detection.add_labels_by_id([o["category_id"]])
        record.detection.add_bboxes([BBox.from_xywh(*o["bbox"])])
        if is_new:
            record.detection.set_scores([o["score"]])
        else:
            record.detection.scores.append(o["score"])

    def template_record(self):
        template = ObjectDetectionRecord()
        template.add_component(ScoresRecordComponent())
        template.add_component(IdmapRecordComponent())
        return template

    def record_id(self, o):
        return o["image_id"]


def test_simple_coco_metric(records, preds):
    simple_coco_metric = SimpleCOCOMetric()
    preds = [Prediction(pred, gt) for pred, gt in zip(preds, records)]
    simple_coco_metric.accumulate(preds)
    result = simple_coco_metric.finalize()
    assert result == 1.0


def test_coco_fake_results():
    class_map = ClassMap(
        [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            None,
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            None,
            "backpack",
            "umbrella",
            None,
            None,
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            None,
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            None,
            "dining table",
            None,
            None,
            "toilet",
            None,
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            None,
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]
    )
    instances_label_file = "/home/ppotrykus/Programs/map_metric_comparison/cocoapi/results/instances_val2014.json"
    fake_results_file = "/home/ppotrykus/Programs/map_metric_comparison/cocoapi/results/instances_val2014_fakebbox100_results.json"
    parser = COCODummyParser(
        annotations_filepath=instances_label_file,
        img_dir="/",
    )
    dets_parser = COCODetectionsParser(
        annotations_filepath=fake_results_file,
        img_dir="/",
        class_map=class_map,
    )
    records, *_ = parser.parse(
        SingleSplitSplitter(), autofix=False, cache_filepath="records"
    )
    preds, *_ = dets_parser.parse(
        SingleSplitSplitter(), autofix=False, cache_filepath="preds"
    )

    # filter records
    records = [
        record
        for record in records
        if record.idmap.get_id(record.record_id) in preds[0].idmap.id2name.values()
    ]
    matched_predictions = []

    for record in records:
        record_true_id = record.idmap.get_id(record.record_id)
        for pred in preds:
            pred_true_id = pred.idmap.get_id(pred.record_id)
            if pred_true_id == record_true_id:
                matched_predictions.append(Prediction(deepcopy(pred), record))

    # coco_metric = COCOMetric(print_summary=True)
    # coco_metric.accumulate(matched_predictions)
    # coco_metric.finalize()

    gt_list, det_list = prediction_to_rodm_lists(matched_predictions)
    result = get_coco_summary(gt_list, det_list)

    simple_coco = SimpleCOCOMetric()
    simple_coco.accumulate(matched_predictions)
    simple_coco.finalize()
