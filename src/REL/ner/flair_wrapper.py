from flair import cache_root
from flair.models import SequenceTagger

from REL.utils import fetch_model


def load_flair_ner(path_or_url):
    try:
        return SequenceTagger.load(path_or_url)
    except Exception:
        raise SystemExit(f"cannot load Flair named entity recognition module from {path_or_url}")
    return SequenceTagger.load(fetch_model(path_or_url, cache_root))
