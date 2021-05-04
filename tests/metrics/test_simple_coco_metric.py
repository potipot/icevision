from icevision.all import *


class COCODummyParser(parsers.COCOBBoxParser):
    def img_size(self, o):
        return ImgSize(0, 0)


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
        record.detection.set_class_map(self.class_map)
        record.detection.add_labels_by_id([o["category_id"]])
        record.detection.add_bboxes([BBox.from_xywh(*o["bbox"])])
        #         pdb.set_trace()
        if is_new:
            record.detection.set_scores([o["score"]])
        else:
            record.detection.scores.append(o["score"])

    def template_record(self):
        template = ObjectDetectionRecord()
        template.add_component(ScoresRecordComponent())
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

    parser = COCODummyParser(
        annotations_filepath="/home/ppotrykus/Datasets/image/coco/annotations/instances_val2014.json",
        img_dir="/",
    )
    dets_parser = COCODetectionsParser(
        annotations_filepath="/home/ppotrykus/Programs/coop/ASR/deepspeech/cocoapi/results/instances_val2014_fakebbox100_results.json",
        img_dir="/",
        class_map=class_map,
    )
    records, *_ = parser.parse(SingleSplitSplitter(), autofix=False)
    preds, *_ = dets_parser.parse(SingleSplitSplitter(), autofix=False)

    def get_true_id(record, parser):
        return parser.idmap.get_id(record.record_id)

    matched_predictions = []

    for record in records:
        record_true_id = get_true_id(record, parser)
        for pred in preds:
            pred_true_id = get_true_id(pred, dets_parser)
            if pred_true_id == record_true_id:
                matched_predictions.append(Prediction(deepcopy(pred), record))

    coco_metric = COCOMetric(print_summary=True)
    coco_metric.accumulate(matched_predictions)
    coco_metric.finalize()

    simple_coco = SimpleCOCOMetric()
    simple_coco.accumulate(matched_predictions)
    simple_coco.finalize()
