from abc import ABC, abstractmethod
from collections import namedtuple

Span = namedtuple("Span", ["text", "start_pos", "end_pos", "score", "tag"])


class NERBase(ABC):
    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError(
            "Base class cannot be used! Subclasses should implement `predict`."
        )
