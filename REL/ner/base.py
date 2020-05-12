from collections import namedtuple

Span = namedtuple("Span", ["text", "start_pos", "end_pos", "score", "tag"])


class NERBase:
    def predict(self, *args, **kwargs):
        raise NotImplementedError(
            "Base class cannot be used! Subclasses should implement `predict`."
        )
