__all__ = ["model"]

import logging
from icevision.imports import *
from icevision.utils import *
from icevision.models.nvidia_nemo.utils import get_model_config

from nemo.collections.asr.models import EncDecClassificationModel
from nemo.core import Exportable
from omegaconf import OmegaConf


class ModelWrapper(EncDecClassificationModel):
    """
    This wrapper is necessary to support explicit kwargs call to forward method which is required by nemo.
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


def list_available_models():
    return ModelWrapper.list_available_models()


def model(
    model_name: str = "commandrecognition_en_matchboxnet3x1x64_v1",
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
        config_file = get_model_config(model_name)
        config = OmegaConf.load(config_file)
        model = ModelWrapper(cfg=config.model)
    if num_classes != 30:
        model.change_labels([""] * num_classes)

    logging.getLogger("nemo_logger").setLevel(original_level)
    return model
