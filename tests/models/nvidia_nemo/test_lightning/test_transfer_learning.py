import pytest
from icevision.all import *


@pytest.fixture
def expected_accuracy():
    return 0.9499


def test_pretrained_default_accuracy(
    google_subset_manifest, expected_accuracy, trainer, pretrained_model
):
    pretrained_model.cfg.test_ds.manifest_filepath = google_subset_manifest.as_posix()
    pretrained_model.setup_test_data(pretrained_model.cfg.test_ds)

    test_results = trainer.test(pretrained_model)
    test_loss, accuracy = first(test_results).values()
    assert abs(accuracy - expected_accuracy) < 1e-3


def test_pretrained_adapter_accuracy(
    audio_single_dataloader, expected_accuracy, trainer, pretrained_model
):
    pl_model = nvidia_nemo.lightning.ModelAdapter(model=pretrained_model)

    test_results = trainer.test(pl_model, audio_single_dataloader)
    test_loss, accuracy = first(test_results).values()
    assert abs(accuracy - expected_accuracy) < 1e-3
