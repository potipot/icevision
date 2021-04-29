import pytest
from icevision.all import *


def test_build_dataset(speech_commands_records):
    train_records, valid_records = speech_commands_records
    dataset = Dataset(train_records)

    sample = first(dataset)
    assert isinstance(sample.wav, torch.Tensor)
    assert sample.wav.shape == torch.Size([16000])
    assert isinstance(sample.img, np.ndarray)
