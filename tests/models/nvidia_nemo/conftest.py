import pytest
from icevision.all import *


@pytest.fixture(params=([0.8, 0.2], [0.5, 0.5]))
def audio_records(google_subset_manifest, gsr_class_map, request):
    audio_parser = parsers.NemoSpeechCommandsParser(
        google_subset_manifest, gsr_class_map, google_subset_manifest.parent
    )
    splits = request.param
    train_records, valid_records = audio_parser.parse(
        data_splitter=RandomSplitter(splits), autofix=False
    )
    return train_records, valid_records


@pytest.fixture
def audio_dataloaders(audio_records):
    train_records, valid_records = audio_records

    train_dataset = Dataset(train_records, tfm=None)
    valid_dataset = Dataset(valid_records, tfm=None)

    train_dl = nvidia_nemo.train_dl(train_dataset, batch_size=3, num_workers=0)
    valid_dl = nvidia_nemo.train_dl(valid_dataset, batch_size=3, num_workers=0)
    return train_dl, valid_dl


@pytest.fixture
def audio_single_dataloader(audio_records):
    train_records, valid_records = audio_records
    dataset = Dataset(train_records + valid_records, tfm=None)
    return nvidia_nemo.train_dl(dataset, batch_size=3, num_workers=0)


@pytest.fixture
def pretrained_model():
    return nvidia_nemo.model(
        # model_name="commandrecognition_en_matchboxnet3x1x64_v1",
        device=torch.device("cpu"),
    )


@pytest.fixture
def scratch_model():
    return nvidia_nemo.model(
        # model_name="commandrecognition_en_matchboxnet3x1x64_v1",
        device=torch.device("cpu"),
        pretrained=False,
    )
