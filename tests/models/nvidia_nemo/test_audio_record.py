from icevision.all import *


def test_speech_commands_record_plotting(audio_records):
    train_records, valid_records = audio_records
    show_record(first(train_records))
    show_records(train_records[:2])
