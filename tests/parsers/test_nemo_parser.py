from fastcore.basics import first
from icevision import SingleSplitSplitter
from icevision.parsers import *


def test_speech_commands_parser(google_subset_manifest, gsr_class_map, samples_source):
    audio_parser = NemoSpeechCommandsParser(
        manifest_filepath=google_subset_manifest,
        class_map=gsr_class_map,
        image_dir=google_subset_manifest.parent,
    )
    records, *_ = audio_parser.parse(data_splitter=SingleSplitSplitter(), autofix=False)
    one_record = first(records)
    one_record.load()

    assert hasattr(one_record, "filepath")
    assert hasattr(one_record, "duration")
    assert len(records) == 20


def test_asr_parser(nemo_asr_manifest, nemo_asr_class_map):
    audio_parser = NemoASRParser(
        manifest_filepath=nemo_asr_manifest,
        class_map=nemo_asr_class_map,
        image_dir=nemo_asr_manifest.parent / "data",
    )
    records, *_ = audio_parser.parse(data_splitter=SingleSplitSplitter(), autofix=False)
    one_record = first(records)
    one_record.load()

    assert hasattr(one_record, "filepath")
    assert hasattr(one_record, "duration")
    assert hasattr(one_record.asr, "text")
    assert hasattr(one_record.asr, "text_encoded")
    assert len(records) == 6
