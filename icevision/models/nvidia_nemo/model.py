__all__ = ["model", "list_available_models"]

import logging

from icevision.imports import *
from icevision.utils import *

from nemo.collections.asr.models import EncDecClassificationModel
from nemo.core import Exportable


class ModelWrapper(EncDecClassificationModel):
    """
    This wrapper is necessary to support explicit kwargs call to forward method which is required by nemo.
    Nemo accuracy is removed due to torch version mismatch (1.8.1 required)
    It also adds a param_groups method required by fastai.
    """

    @torch.jit.export
    def forward(self, input_signal, input_signal_length, labels=None, labels_len=None):
        """Verbatim copy of  nemo classification models forward, only without typecheck decorator"""
        processed_signal, processed_signal_len = self.preprocessor(
            input_signal=input_signal,
            length=input_signal_length,
        )
        # Crop or pad is always applied
        if self.crop_or_pad is not None:
            processed_signal, processed_signal_len = self.crop_or_pad(
                input_signal=processed_signal, length=processed_signal_len
            )
        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal)
        encoded, encoded_len = self.encoder(
            audio_signal=processed_signal, length=processed_signal_len
        )
        logits = self.decoder(encoder_output=encoded)
        return logits

    def param_groups(self) -> List[List[nn.Parameter]]:
        layers = [
            self.encoder,
            self.decoder,
        ]
        param_groups = [list(layer.parameters()) for layer in layers]
        check_all_model_params_in_groups2(self, param_groups)
        return param_groups

    def export(self, *args, **kwargs):
        return Exportable.export(self, *args, **kwargs)

    @property
    def num_weights(self):
        return -1

    @classmethod
    def from_scratch(cls, model_name):
        cfg = cls.from_pretrained(model_name=model_name).to_config_dict()
        cfg.test_ds.manifest_filepath = None
        cfg.validation_ds.manifest_filepath = None
        cfg.train_ds.manifest_filepath = None
        return cls.from_config_dict(cfg)


def model(
    model_name: str = "MatchboxNet-3x2x64-v1",
    num_classes: int = 30,
    # TODO: img_size: int,
    pretrained: bool = True,
    device: Optional["torch.device"] = None,
) -> nn.Module:
    """Check the available pretrained models with `nemo_asr.list_available_models()`"""
    original_level = logging.getLogger("nemo_logger").getEffectiveLevel()
    logging.getLogger("nemo_logger").setLevel(logging.ERROR)

    if pretrained == True:
        model = ModelWrapper.from_pretrained(model_name=model_name, map_location=device)
    else:
        model = ModelWrapper.from_scratch(model_name=model_name)
    if num_classes != 30:
        model.change_labels([""] * num_classes)

    logging.getLogger("nemo_logger").setLevel(original_level)
    return model


def list_available_models():
    return ModelWrapper.list_available_models()
