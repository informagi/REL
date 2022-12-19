import sys

from REL.entity_disambiguation import EntityDisambiguation
from REL.mention_detection import MentionDetection
from REL.utils import process_results


class REL_ED:
    def __init__(self, base_url, wiki_version):

        config = {
            "mode": "eval",
            "model_path": f"{base_url}/{wiki_version}/generated/model",
        }

        self.mention_detection = MentionDetection(
            base_url, wiki_version
        )  # This is only used for format spans
        self.model = EntityDisambiguation(base_url, wiki_version, config)

    def generate_response(self, text, spans):
        """Generate ED results

        Returns:
            - list of tuples for each entity found.

        Note:
            - Original code: https://github.com/informagi/REL/blob/9ca253b1d371966c39219ed672f39784fd833d8d/REL/server.py#L101
        """

        API_DOC = "API_DOC"

        if len(text) == 0 or len(spans) == 0:
            return []

        # Get the mentions from the spans
        processed = {API_DOC: [text, spans]}
        mentions_dataset, total_ment = self.mention_detection.format_spans(processed)

        # Disambiguation
        predictions, timing = self.model.predict(mentions_dataset)

        # Process result.
        result = process_results(
            mentions_dataset,
            predictions,
            processed,
            include_offset=False if (len(spans) > 0) else True,
        )

        # Singular document.
        if len(result) > 0:
            return [*result.values()][0]

        return []

    def ed(self, text, spans):
        """Change tuple to list to match the output format of REL API."""
        response = self.generate_response(text, spans)
        return [list(ent) for ent in response]
