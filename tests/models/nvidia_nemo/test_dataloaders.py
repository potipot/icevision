import pytest
from icevision.all import *


def test_fail_build_train_batch(audio_records):
    # should fail cause wasn't wrapped in Dataset and audio file was not loaded in getitem
    train_records, _ = audio_records
    with pytest.raises(AttributeError) as excinfo:
        (waves, targets), records = nvidia_nemo.build_train_batch(train_records)
    assert first(excinfo.value.args) == "BaseRecord has no attribute wav"


def test_nemo_asr_train_dl(audio_single_dataloader):
    train_dl = audio_single_dataloader
    batch = first(train_dl)
