from icevision.all import *


def test_speech_commands_record_plotting(speech_commands_records):
    train_records, valid_records = speech_commands_records
    show_record(first(train_records))
    show_records(train_records[:2])


def test_asr_record_plotting(asr_records):
    train_records, valid_records = asr_records
    show_record(first(train_records))
    show_records(train_records[:2])
