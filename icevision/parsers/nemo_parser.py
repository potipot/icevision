__all__ = [
    "NemoASRParser",
    "NemoSpeechCommandsParser",
]

from icevision.imports import *
from icevision.core import *
from icevision.parsers import *


class NemoBaseParser(Parser):
    def __init__(
        self,
        manifest_filepath: Union[Path, str],
        class_map: ClassMap,
        image_dir: Optional[Union[Path, str]] = "",
    ):
        super().__init__(template_record=self.template_record())
        self.manifest_filepath = manifest_filepath
        self.image_dir = Path(image_dir)
        self.class_map = class_map

    def __iter__(self) -> Any:
        with open(self.manifest_filepath) as manifest:
            for line in manifest.readlines():
                yield json.loads(line)

    def record_id(self, o) -> Hashable:
        return hash(o["audio_filepath"])

    @abstractmethod
    def template_record(self):
        raise NotImplementedError

    def parse_fields(self, o, record: BaseRecord, is_new: bool) -> None:
        record.set_filepath(self.image_dir / o["audio_filepath"])
        record.set_duration(o["duration"])


class NemoSpeechCommandsParser(NemoBaseParser):
    def parse_fields(self, o, record, is_new):
        super().parse_fields(o, record, is_new)
        record.classification.set_class_map(self.class_map)
        record.classification.add_labels([o["command"]])

    def template_record(self):
        return BaseRecord(
            (
                WaveformRecordComponent(),
                WaveformFilepathRecordComponent(),
                ClassificationLabelsRecordComponent(),
            )
        )


class NemoASRParser(NemoBaseParser):
    def parse_fields(self, o, record, is_new):
        super().parse_fields(o, record, is_new)
        record.asr.set_class_map(self.class_map)
        record.asr.set_text(o["text"])

    def template_record(self):
        return BaseRecord(
            (
                WaveformRecordComponent(),
                WaveformFilepathRecordComponent(),
                TextlabelsRecordComponent(),
            )
        )
