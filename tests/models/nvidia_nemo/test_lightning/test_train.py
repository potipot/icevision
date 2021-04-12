import pytest
from icevision.all import *
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def test_pretrained_no_data(pretrained_model, trainer):
    pl_model = nvidia_nemo.lightning.ModelAdapter(model=pretrained_model)
    with pytest.raises(MisconfigurationException) as misconfig:
        trainer.fit(pl_model)
    assert "No `train_dataloader()` method defined." in first(misconfig.value.args)


# here parametrizing with fixture
@pytest.mark.parametrize("model", ["scratch_model", "pretrained_model"])
def test_train(model, audio_dataloaders, trainer, request):
    model = request.getfixturevalue(model)
    train_dl, valid_dl = audio_dataloaders

    class LightModel(nvidia_nemo.lightning.ModelAdapter):
        def configure_optimizers(self):
            return SGD(self.parameters(), lr=1e-4)

    light_model = LightModel(model=model)
    trainer.fit(light_model, train_dl, valid_dl)
