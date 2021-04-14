__all__ = ["ModelAdapter"]

from icevision.imports import *
from icevision.metrics import *
from icevision.engines.lightning.lightning_model_adapter import LightningModelAdapter

from nemo.collections.asr.models import EncDecClassificationModel


class ModelAdapter(LightningModelAdapter, ABC):
    """Lightning module specialized for nemo classification. This class kinda mimics fastai's learner object

    The methods `forward`, `training_step`, `validation_step`, `validation_epoch_end`
    are already overriden.

    # Arguments
        model: The pytorch model to use.
        metrics: `Sequence` of metrics to use.

    # Returns
        A `LightningModule`.
    """

    def __init__(self, model: nn.Module, metrics: List[Metric] = None):
        super().__init__(metrics=metrics)
        self.model = model
        self.validation_accuracy = pl.metrics.Accuracy()
        self.training_accuracy = pl.metrics.Accuracy()

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        (audio_signal, input_signal_length, labels, labels_len), records = batch
        logits = self.forward(
            input_signal=audio_signal, input_signal_length=input_signal_length
        )

        loss = self.model.loss(logits=logits, labels=labels)
        self.training_accuracy(preds=logits, target=labels)
        self.log("training_loss", loss.detach())
        self.log("training_accuracy", self.training_accuracy)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        (audio_signal, audio_signal_len, labels, labels_len), records = batch
        logits = self.forward(
            input_signal=audio_signal, input_signal_length=audio_signal_len
        )

        validation_loss = self.model.loss(logits=logits, labels=labels)
        self.validation_accuracy(preds=logits, target=labels)
        self.log("validation_loss", validation_loss)
        self.log("validation_accuracy", self.validation_accuracy)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        (audio_signal, audio_signal_len, labels, labels_len), records = batch
        logits = self.forward(
            input_signal=audio_signal, input_signal_length=audio_signal_len
        )

        test_loss = self.model.loss(logits=logits, labels=labels)
        self.validation_accuracy(preds=logits, target=labels)
        self.log("test_loss", test_loss)
        self.log("test_accuracy", self.validation_accuracy)
