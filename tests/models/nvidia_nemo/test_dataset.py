import pytest
from icevision.all import *


def test_build_dataset(audio_records):
    train_records, valid_records = audio_records
    dataset = Dataset(train_records)

    sample = first(dataset)
    assert isinstance(sample.wav, torch.Tensor)
    assert sample.wav.shape == (1, 16000)
    assert isinstance(sample.img, np.ndarray)
