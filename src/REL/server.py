from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import load_flair_ner
from flair.models import SequenceTagger
from REL.mention_detection import MentionDetection
from REL.utils import process_results


class ModelHandler:
    API_DOC = "API_DOC"

    def __init__(self, base_url, wiki_version, ed_model, ner_model):
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
                          conversation: bool = False):
        """
        Generates response for API. Can be either ED only or EL, meaning end-to-end.

        :return: list of tuples for each entity found.
        """

        if len(text) == 0:
            return []

        if len(spans) > 0:
            # ED.
            processed = {self.API_DOC: [text, spans]}
            mentions_dataset, total_ment = self.mention_detection.format_spans(
                processed)
        else:
            # EL
            processed = {self.API_DOC: [text, spans]}
            mentions_dataset, total_ment = self.mention_detection.find_mentions(
                processed, self.tagger_ner)

        # Disambiguation
        predictions, timing = self.model.predict(mentions_dataset)

        # Process result.
        result = process_results(
            mentions_dataset,
            predictions,
            processed,
            include_offset=False if
            ((len(spans) > 0) or self.custom_ner) else True,
        )

        # Singular document.
        if len(result) > 0:
            return [*result.values()][0]

        return []


from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

@app.get("/")
def root():
    """Returns server status."""
    return {
        "schemaVersion": 1,
        "label": "status",
        "message": "up",
        "color": "green",
    }


class EntityConfig(BaseModel):
    text: str = Field(..., description="Text for entity linking or disambiguation.")
    spans: List[str] = Field(..., description="Spans for entity disambiguation.")


@app.post("/")
def root(config: EntityConfig):
    """Submit your text here for entity disambiguation or linking."""
    response = handler.generate_response(text=config.text, spans=config.spans)
    return response


if __name__ == "__main__":
    import argparse
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("base_url")
    p.add_argument("wiki_version")
    p.add_argument("--ed-model", default="ed-wiki-2019")
    p.add_argument("--ner-model", default="ner-fast")
    p.add_argument("--bind", "-b", metavar="ADDRESS", default="0.0.0.0")
    p.add_argument("--port", "-p", default=5555, type=int)
    args = p.parse_args()

    ner_model = load_flair_ner(args.ner_model)
    ed_model = EntityDisambiguation(
        args.base_url, args.wiki_version, {"mode": "eval", "model_path": args.ed_model}
    )

    handler = ModelHandler(args.base_url, args.wiki_version, ed_model, ner_model)

    uvicorn.run(app, port=args.port, host=args.bind)
