from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import load_flair_ner
from flair.models import SequenceTagger
from REL.mention_detection import MentionDetection
from REL.utils import process_results


class ResponseModel:
    API_DOC = "API_DOC"

    def __init__(self, base_url, wiki_version, model, tagger_ner=None):
        self.model = model
        self.tagger_ner = tagger_ner

        self.base_url = base_url
        self.wiki_version = wiki_version

        self.custom_ner = not isinstance(tagger_ner, SequenceTagger)
        self.mention_detection = MentionDetection(base_url, wiki_version)

    def generate_response(self,
                          *,
                          text: list,
                          spans: list,
                          ):
        """
        Generates response for API. Can be either ED only or EL, meaning end-to-end.

        :return: list of tuples for each entity found.
        """

        if len(text) == 0:
            return []
        
        processed = {self.API_DOC: [text, spans]}

        if len(spans) > 0:
            # ED.
            mentions_dataset, total_ment = self.mention_detection.format_spans(
                processed
            )
        else:
            # EL
            mentions_dataset, total_ment = self.mention_detection.find_mentions(
                processed, self.tagger_ner
            )

        # Disambiguation
        predictions, timing = self.model.predict(mentions_dataset)

        include_offset = (len(spans) == 0) and not self.custom_ner

        # Process result.
        result = process_results(
            mentions_dataset,
            predictions,
            processed,
            include_offset=include_offset,
        )

        # Singular document.
        if len(result) > 0:
            return [*result.values()][0]

        return []