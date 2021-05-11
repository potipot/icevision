__all__ = ["SimpleCOCOMetric"]

import pdb

from icevision.data.prediction import Prediction
from icevision.metrics.metric import Metric
from icevision.metrics.confusion_matrix.confusion_matrix_utils import *
from icevision.imports import *
import pandas as pd
from scipy import interpolate
import pdb


def build_results_table(single_label_result, name=""):
    is_correct_labels = []
    iou_scores = []
    scores = []
    for prediction_item in single_label_result:
        match = get_best_iou_match(prediction_item.matches)
        is_correct_label = prediction_item.label_id == match.label_id
        iou_score = match.iou_score
        score = prediction_item.score
        is_correct_labels.append(is_correct_label)
        iou_scores.append(iou_score)
        scores.append(score)

    df = pd.DataFrame.from_dict(
        dict(is_correct_label=is_correct_labels, iou_score=iou_scores, score=scores)
    )
    df.name = name
    return df


def auc(x: pd.Series, y: pd.Series) -> float:
    """Calculate area under curve set by x,y points using Riemann right endpoint approximation"""
    x_increment = x.diff().fillna(x.iloc[0])
    return np.dot(y, x_increment)


def auc_trapz(x, y) -> float:
    """Calculate area under curve set by x,y points using trapezoidal approximation.
    Inserting duplicate row starting from (y[0], 0.0) to approximate correctly and not from (0.0, 0.0)"""
    return np.trapz([y.iloc[0], *y], [0.0, *x])


def auc_101_linear_interp(x: pd.Series, y: pd.Series) -> float:
    xw = np.linspace(0, 1, 101)
    y101 = np.interp(xw, xp=x, fp=y)
    x101 = np.full(101, 0.01)
    return np.dot(x101, y101)


def auc_101_next_interp(x: pd.Series, y: pd.Series) -> float:
    # fig, ax = plt.subplots()
    # ax.scatter(x, y, marker="x")
    last_precision = y.iloc[-1] if x.iloc[-1] == 1.0 else 0.0
    option = 2

    # option 1 include (0.0, y[0]) and (1.0, y[-1]) points
    if option == 1:
        x = [0.0, *x, 1.0]
        y = [y.iloc[0], *y, last_precision]
    # option 2 include (1.0, y[-1]) point
    elif option == 2:
        x = [*x, 1.0]
        y = [*y, last_precision]
    # option 3 dont add any points
    elif option == 3:
        x = [*x]
        y = [*y]
        # nasty hack for len(x) = 1
        if len(x) == 1:
            x = x * 2
            y = y * 2
    interp_func = interpolate.interp1d(
        x, y, fill_value=(y[0], y[-1]), bounds_error=False, kind="next"
    )
    xw = np.linspace(0, 1, 101)
    x101 = np.full(101, 0.01)
    y101 = interp_func(xw)
    # ax.scatter(xw, y101, marker=".")
    # ax.set_ylim([0.0, 1.0])
    # pdb.set_trace()
    # return np.mean(y101)
    return np.dot(x101, y101)


class SimpleCOCOMetric(Metric):
    def __init__(self):
        super().__init__()
        self.matches = []
        self.class_map = None
        self.gt_counter = Counter()

    def _reset(self):
        self.matches = []

    def accumulate(self, preds):
        for pred in preds:
            target_record = pred.ground_truth
            prediction_record = pred.pred
            self.gt_counter.update(target_record.detection.labels)
            self.class_map = target_record.detection.class_map
            matches = match_targets_to_predictions(
                target=target_record, prediction=prediction_record
            )
            self.matches.extend(matches)

    def finalize(self):
        # main loop over classes
        APs = []
        # iterate over ground truths only in self.gt_detection_counter to skip classes non-existent in GTS
        for label_name in self.gt_counter.keys():
            label_id = self.class_map.get_by_name(label_name)
            if label_id == 0:
                raise RuntimeWarning(
                    "Using label_id = 0 for ground truths, will lead to wrong mAP results, averaging with backgrounds"
                )
                # TODO: ignore background in mAP average calculation
            single_label_result = filter_by_label_id(label_id, self.matches)
            single_label_result = sorted(
                single_label_result, key=lambda item: item.score, reverse=True
            )
            results_table = build_results_table(single_label_result, name=label_id)
            iou_threshold = 0.5
            APs.append(
                self.calculate_ap(
                    results_table,
                    iou_threshold,
                    n_ground_truths=self.gt_counter[label_name],
                )
            )
        return np.mean(APs)

    @staticmethod
    def calculate_ap(df, iou_threshold, n_ground_truths):
        # FIXME: > or >= for iou_threshold?
        # df.sort_values(by="score", ascending=False, inplace=True, ignore_index=True)
        df["is_true_positive"] = df.is_correct_label * df.iou_score > iou_threshold
        df["n_tps"] = df.is_true_positive.cumsum()
        # index to start from 1, necessary for precision calculation
        df.index += 1
        df["precision"] = df.n_tps.to_frame().apply(
            lambda row: row.n_tps / row.name, axis=1
        )
        df["recall"] = df.n_tps / n_ground_truths
        padilla_results = pd.read_csv(
            f"/home/ppotrykus/Programs/map_metric_comparison/tests/outputs/{df.name}"
        )
        # get back to original indexing
        df.reset_index(inplace=True, drop=True)
        assert all(padilla_results["recall"] == df["recall"])
        assert all(padilla_results["precision"] == df["precision"])
        if not df.empty:
            # picking the best precision for a given recall to avoid zig-zag pattern
            df["precision_interpolated"] = df.apply(
                lambda row: df.precision[df.recall >= row.recall].max(), axis=1
            )
            # ap = auc(x=df.recall, y=df.precision_interpolated)
            # ap = auc_101_linear_interp(x=df.recall, y=df.precision_interpolated)
            ap = auc_101_next_interp(x=df.recall, y=df.precision_interpolated)
            # fig = plt.figure()
            # x = np.array(df.recall)
            # y = np.array(df.precision_interpolated)
            # ax = fig.add_subplot(111)
            # ax.scatter(x, y)
            # ax.set_xlim([0.0, 1.0])
            # ax.set_ylim([0.0, 1.0])
        else:
            ap = 0
        # print(df)
        return ap
