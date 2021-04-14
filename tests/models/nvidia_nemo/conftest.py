import pytest
from icevision.all import *


@pytest.fixture(params=([0.8, 0.2], [0.5, 0.5]))
def speech_commands_records(google_subset_manifest, gsr_class_map, request):
    audio_parser = parsers.NemoSpeechCommandsParser(
        manifest_filepath=google_subset_manifest,
        class_map=gsr_class_map,
        image_dir=google_subset_manifest.parent,
    )
    splits = request.param
    train_records, valid_records = audio_parser.parse(
        data_splitter=RandomSplitter(splits), autofix=False
    )
    return train_records, valid_records


@pytest.fixture(params=([0.8, 0.2], [0.5, 0.5]))
def asr_records(nemo_asr_manifest, nemo_asr_class_map, request):
    audio_parser = parsers.NemoASRParser(
        manifest_filepath=nemo_asr_manifest,
        class_map=nemo_asr_class_map,
        image_dir=nemo_asr_manifest.parent / "data",
    )
    splits = request.param
    train_records, valid_records = audio_parser.parse(
        data_splitter=RandomSplitter(splits), autofix=False
    )
    return train_records, valid_records


@pytest.fixture
def audio_dataloaders(speech_commands_records):
    train_records, valid_records = speech_commands_records

    train_dataset = Dataset(train_records, tfm=None)
    valid_dataset = Dataset(valid_records, tfm=None)

    train_dl = nvidia_nemo.train_dl(train_dataset, batch_size=3, num_workers=0)
    valid_dl = nvidia_nemo.train_dl(valid_dataset, batch_size=3, num_workers=0)
    return train_dl, valid_dl


@pytest.fixture
def audio_single_dataloader(speech_commands_records):
    train_records, valid_records = speech_commands_records
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
