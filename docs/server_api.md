# REL server API docs

This page documents usage for the [REL server](https://rel.cs.ru.nl/docs). The live, up-to-date api can be found either [here](https://rel.cs.ru.nl/api/docs) or [here](https://rel.cs.ru.nl/api/redocs).

Scroll down for code samples, example requests and responses.

## Server status

`GET /`

Returns server status.

### Example

> Response

```json
{
  "schemaVersion": 1,
  "label": "status",
  "message": "up",
  "color": "green"
}
```

> Code

```python
>>> import requests
>>> requests.get("https://rel.cs.ru.nl/api/").json()
{'schemaVersion': 1, 'label': 'status', 'message': 'up', 'color': 'green'}
```

## Named Entity Linking

`POST /`  
`POST /ne`  

Submit your text here for entity disambiguation or linking.

The REL annotation mode can be selected by changing the path.
use `/` or `/ne` for annotating regular text with named
entities (default), `/ne_concept` for regular text with both concepts and
named entities, and `/conv` for conversations with both concepts and
named entities.

> Schema 

`text` (string)
: Text for entity linking or disambiguation.

`spans` (list)

: For EL: the spans field needs to be set to an empty list.

: For ED: spans should consist of a list of tuples, where each tuple refers to the start position (int) and length of a mention (int).

: This is used when mentions are already identified and disambiguation is only needed. Each tuple represents start position and length of mention (in characters); e.g., [(0, 8), (15,11)] for mentions 'Nijmegen' and 'Netherlands' in text 'Nijmegen is in the Netherlands'.

`tagger` (string)
: NER tagger to use. Must be one of `ner-fast`, `ner-fast-with-lowercase`. Default: `ner-fast`.

### Example

> Request body

```json
{
  "text": "If you're going to try, go all the way - Charles Bukowski.",
  "spans": [
    [
      41,
      16
    ]
  ],
  "tagger": "ner-fast"
}
```

> Response

The 7 values of the annotation represent the start index, end index, the annotated word, prediction, ED confidence, MD confidence, and tag.

```json
[

    [
        41,
        16,
        "Charles Bukowski",
        "Charles_Bukowski",
        0,
        0,
        "NULL"
    ]

]
```

> Code

```python
>>> import requests
>>> myjson = {
  "text": "REL is a modular Entity Linking package that can both be integrated in existing pipelines or be used as an API.",
  "tagger": "ner-fast"
}
>>> requests.post("https://rel.cs.ru.nl/api/ne", json=myjson).json()
[[0, 3, 'REL', 'Category_of_relations', 0, 0, 'ORG'], [107, 3, 'API', 'Application_programming_interface', 0, 0, 'MISC']]
```

## Conversational entity linking

`POST /conv`

Submit your text here for conversational entity linking.

> Schema

`text` (list)
: Text is specified as a list of turns between two speakers.

    `speaker` (string)
    : Speaker for this turn, must be one of `USER` or `SYSTEM`

    `utterance` (string)
    : Input utterance to be annotated.

`tagger` (string)
: NER tagger to use. Choices: `default`.


### Example

> Request body

```json
{
  "text": [
    {
      "speaker": "USER",
      "utterance": "I am allergic to tomatoes but we have a lot of famous Italian restaurants here in London."
    },
    {
      "speaker": "SYSTEM",
      "utterance": "Some people are allergic to histamine in tomatoes."
    },
    {
      "speaker": "USER",
      "utterance": "Talking of food, can you recommend me a restaurant in my city for our anniversary?"
    }
  ],
  "tagger": "default"
}
```

> Response

The 7 values of the annotation represent the start index, end index, the annotated word, prediction, ED confidence, MD confidence, and tag.

```json
[
  {
    "speaker": "USER",
    "utterance": "I am allergic to tomatoes but we have a lot of famous Italian restaurants here in London.",
    "annotations": [
      [17, 8, "tomatoes", "Tomato"],
      [54, 19, "Italian restaurants", "Italian_cuisine"],
      [82, 6, "London", "London"]
    ]
  },
  ...
]
```

> Code

```python
>>> import requests
>>> myjson = {
  "text": [...],
  "tagger": "default"
}
>>> requests.post("https://rel.cs.ru.nl/api/conv", json=myjson).json()
[{...}]
```


## Conceptual entity linking

`POST /ne_concept`

Submit your text here for conceptual entity disambiguation or linking.

### Example

> Request body

```json
{}
```

> Response

Not implemented.

```json
{}
```

> Code

```python
>>> import requests
>>> myjson = {
  "text": ...,
}
>>> requests.post("https://rel.cs.ru.nl/api/ne_concept", json=myjson).json()
{...}
```

