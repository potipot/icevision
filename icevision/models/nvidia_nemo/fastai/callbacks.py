__all__ = ["NemoCallback"]

from icevision.imports import *
from icevision.engines.fastai import *


class NemoCallback(fastai.Callback):
    def before_batch(self):
        assert len(self.xb) == len(self.yb) == 1, "Only works for single input-output"
        self.learn.records = self.yb[0]
        self.learn.xb = self.xb[0]
        # fooling fastai to not calculate loss automatically
        self.learn.yb = ()

    def after_pred(self):
        logits = self.learn.pred
        audio_signal, audio_signal_len, labels, labels_len = self.xb
        self.learn.loss = self.loss_func(logits=logits, labels=labels)
        self.learn.yb = ()
        # self.learn.pred = logits

        # if not self.training:
        # preds = efficientdet.convert_raw_predictions(
        #     self.pred["detections"], self.learn.records, 0
        # )
        #
        # preds = self.learn.pred
        # self.learn.converted_preds = preds

    def after_loss(self):
        # need to assign learn.yb here, cause automatic loss doesnt work, nemo requires explicit kwargs assignment
        audio_signal, audio_signal_len, labels, labels_len = self.xb
        self.learn.yb = [labels]
