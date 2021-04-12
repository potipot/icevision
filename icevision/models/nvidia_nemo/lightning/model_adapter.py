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

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        audio_batch, records = batch
        input_signal, input_signal_length, labels, labels_len = audio_batch

        logits = self.forward(
            input_signal=input_signal, input_signal_length=input_signal_length
        )
        loss_value = self.model.loss(logits=logits, labels=labels)

        logs = {"loss": loss_value}

        # TODO: move metrics outside
        self.model._accuracy.update(logits=logits, labels=labels)
        top_k = self.model._accuracy.compute()
        for i, top_i in enumerate(top_k):
            logs[f"training_batch_accuracy_top@{i}"] = top_i

        self.log_dict(logs)
        return logs

    # TODO: finish this for DDP2
    # def training_step_end(self, batch_parts):
    #     gpu_0_prediction = batch_parts[0]['pred']
    #     gpu_1_prediction = batch_parts[1]['pred']
    #
    #     # do something with both outputs
    #     return (batch_parts[0]['loss'] + batch_parts[1]['loss']) / 2

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        audio_batch, records = batch
        logs = self.model.validation_step(
            batch=audio_batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx
        )
        return logs

    def validation_step_end(self, batch_parts):
        val_loss = batch_parts["val_loss"]
        val_accuracy = first(batch_parts["val_acc"])
        logs = {"val_loss": val_loss, "val_accuracy": val_accuracy}
        self.log_dict(logs)
        return logs

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        audio_batch, records = batch
        logs = self.model.test_step(
            batch=audio_batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx
        )
        return logs

    def test_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self.model.multi_test_epoch_end(outputs, dataloader_idx)
