from REL.utils import ModelLoader
from flair import cache_root
from flair.models import SequenceTagger


class FlairNERWrapper:
    def load(path_or_url):
        try:
            return SequenceTagger.load(path_or_url)
        except FileNotFoundError:
            pass
        return SequenceTagger.load(ModelLoader.fetch(path_or_url, cache_root))
