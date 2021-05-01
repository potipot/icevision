from icevision.all import *


def test_simple_coco_metric(records, preds):
    simple_coco_metric = SimpleCOCOMetric()
    preds = [Prediction(pred, gt) for pred, gt in zip(preds, records)]
    simple_coco_metric.accumulate(preds)
    result = simple_coco_metric.finalize()
    assert result == 1.0
