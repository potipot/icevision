__all__ = ["ClassMap", "BACKGROUND"]

from icevision.imports import *

BACKGROUND = "background"


class ClassMap:
    """Utility class for mapping between class name and id."""

    def __init__(self, classes: List[str], background: Optional[int] = 0):
        classes = classes.copy()

        if background is not None:
            if background == -1:
                background = len(classes)
            classes.insert(background, "background")

        self.id2class = classes
        self.class2id = {name: i for i, name in enumerate(classes)}

        self._lock = True

    def __len__(self):
        return len(self.id2class)

    def get_id(self, id: int) -> str:
        return self.id2class[id]

    def get_name(self, name: str) -> int:
        try:
            return self.class2id[name]
        except KeyError as e:
            if not self._lock:
                self.id2class.append(name)
                self.class2id[name] = len(self.class2id)
            else:
                raise e

    def lock(self):
        self._lock = True

    def unlock(self):
        self._lock = False

    def set_background(self, id: int):
        if BACKGROUND in self.id2class:
            raise RuntimeError(
                "Background should not be present in original class_map"
                "as it's dependent on the model"
            )
        self.id2class.insert(id, BACKGROUND)
        self.class2id = {name: i for i, name in enumerate(self.id2class)}

    def __repr__(self):
        return f"<ClassMap: {self.class2id.__repr__()}>"
