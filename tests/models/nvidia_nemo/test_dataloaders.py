import pytest
from icevision.all import *


def test_fail_build_train_batch(speech_commands_records):
    # should fail cause wasn't wrapped in Dataset and audio file was not loaded in getitem, therefore shape is invalid
    train_records, _ = speech_commands_records
    with pytest.raises(AttributeError) as excinfo:
        (waves, targets), records = nvidia_nemo.build_train_batch(train_records)
    assert first(excinfo.value.args) == "'NoneType' object has no attribute 'shape'"


def test_nemo_asr_train_dl(audio_single_dataloader):
    train_dl = audio_single_dataloader
    batch = first(train_dl)


def test_nemo_asr_show_batch(audio_single_dataloader):
    train_dl = audio_single_dataloader
    batch = first(train_dl)
    nvidia_nemo.show_batch(batch)
