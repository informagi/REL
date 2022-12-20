from REL.response_model import ResponseModel

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

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


class ConversationTurn(BaseModel):
    speaker: Literal["USER", "SYSTEM"] = Field(..., description="Speaker for this turn.")
    utterance: str = Field(..., description="Input utterance.")


class ConversationConfig(BaseModel):
    text: List[ConversationTurn] = Field(..., description="Conversation as list of turns between two speakers.")


@app.post("/conversation/")
def conversation(config: ConversationConfig):
    """Submit your text here for conversational entity linking."""
    text = config.dict()['text']
    response = conv_handler.annotate(text)
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

    from REL.crel.conv_el import ConvEL
    from REL.entity_disambiguation import EntityDisambiguation
    from REL.ner import load_flair_ner

    ner_model = load_flair_ner(args.ner_model)
    ed_model = EntityDisambiguation(
        args.base_url, args.wiki_version, {"mode": "eval", "model_path": args.ed_model}
    )

    handler = ResponseModel(args.base_url, args.wiki_version, ed_model, ner_model)

    conv_handler = ConvEL(args.base_url, args.wiki_version, ed_model=ed_model)

    uvicorn.run(app, port=args.port, host=args.bind)
