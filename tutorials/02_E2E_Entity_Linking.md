# End-to-End Entity Linking
In this tutorial we will guide you through the process of using our Entity Linking system. Please take note that we assume
that the tutorials are followed in a sequential order, meaning that the variable `base_url` is defined and that you have
amended the required project structure.


## Setting up  your own API
Previously we defined our `base_url`. We also need the project to know which specific
Wikipedia corpus we would like to use. We do this by creating a variable that in this case refers to the folder containing
the necessary files for our Wikipedia 2014 folder. Additionally, we import the required packages.
 
 ```python
from http.server import HTTPServer
from flair.models import SequenceTagger

from REL.entity_disambiguation import EntityDisambiguation
from REL.server import make_handler

wiki_subfolder = "wiki_2014"
 ```

Now that we know which Wikipedia corpus we are using, we need to create a configuration dictionary for our Entity
Disambiguation model, after which we may instantiate the model. To reduce the load of our model, we choose to load
all necessary files such as the embeddings into a sqlite3 database. Per document that the model is fed, we perform
a lookup to the embeddings that are required to come up with the result. During our predictions we noticed that many
of these mentions had overlapping entities, thus making it inefficient to constantly remove them from memory. As such,
we choose to keep them in memory, but the user may choose to remove embeddings after prediction if he wishes to do so
by setting `reset_embeddings=True`.

```python
config = {
    "mode": "eval",
    "model_path": "{}/{}/generated/model".format(base_url, wiki_subfolder),
}

model = EntityDisambiguation(base_url, wiki_subfolder, config, reset_embeddings=False)
```

As was mentioned prior to this, we used Flair's NER tagger as our Mention Detection system. This system can be replaced,
which we elaborate on in the sections below. For now we assume that you also want to use their NER tagger (it's awesome).
Take note that we used their fast tagger for our predictions as we wanted to use a light-weight model that could serve
users in terms of speed.

```python
tagger_ner = SequenceTagger.load("ner-fast")
```

Our final step consists of starting the server, where we may define our IP-address, port and most importantly the `MODE`
that we would like to predict in. The available modes are either `EL` or `ED`. In production a user will only want to use
the `EL` mode, even when using their own MD system. We only used the `ED` mode for GERBIL evaluation. Additionally,
the user may choose to include or exclude confidence scores.

```python
MODE = "EL"
INCLUDE_CONF = True

server_address = ("localhost", 5555)
server = HTTPServer(
    server_address,
    make_handler(
        base_url, wiki_subfolder, model, tagger_ner, mode=MODE, include_conf=INCLUDE_CONF
    ),
)

try:
    print("Ready for listening.")
    server.serve_forever()
except KeyboardInterrupt:
    exit(0)
```

Once the server is ready for listening, one or more users may query it for results. Querying the API is straightforward and
we have added an example below. We believe such an API is especially useful for research groups that want to run a single
server with our package and query it with multiple people. A user may query the API using the code below, where the "text" 
field may be replaced with a text for a particular document. At this point in time, a user may only query the server with
a single document. We currently added this to make sure that a single user does not overload a particular server.

```python
import requests

document = {
    "text": """If you're going to try, go all the way - Charles Bukowski""",
    "spans": [],
}

API_result = requests.post("http://localhost:5555", json=document).json()
```

## Pipeline integration
Alternatively, one may choose to integrate the package into an existing pipeline. This grants the user a bit more freedom
with inputting multiple documents at the same time. This can especially be useful when batch loading multiple documents
using their own or our Mention Detection system. To do this we once more import the required packages and define the folder name that contains our Wikipedia corpus files.

 ```python
from flair.models import SequenceTagger

from REL.mention_detection import MentionDetection
from REL.utils import process_results
from REL.entity_disambiguation import EntityDisambiguation

wiki_subfolder = "wiki_2014"
 ```

The code below serves as an example as to how users should format their dataset. This should obviously be replaced
and should not be used in production, although the outcome is the `input` variable that will be used throughout this
tutorial.

```python
def example_preprocessing():
    # user does some stuff, which results in the format below.
    text = "Obama will visit Germany. And have a meeting with Merkel tomorrow."
    processed = {"test_doc1": [text, []], "test_doc2": [text, []]}
    return processed

input = example_preprocessing()
```

Now that we have defined our `input` we instantiate our mention detection class, NER-tagger and use both to find
mentions of our dataset. The output of the function `find_mentions()` is a dictionary with various properties that
are required for the Entity Disambiguation module. Additionally, it returns a count consisting of the total number of
mentions that were found using the NER-tagger. This number may be unequal to the number of mentions that the dictionary
`mentions_dataset` stores as this dictionary only stores mentions that were found in our Knowledge Base.

```python
mention_detection = MentionDetection(base_url, wiki_subfolder)
tagger_ner = SequenceTagger.load("ner-fast")
mentions_dataset, n_mentions = mention_detection.find_mentions(input, tagger_ner)
```

The final step of our End-to-End process consists of instantiating the model and using it to find entities based
on the mentions that we previously found.

```python
config = {
    "mode": "eval",
    "model_path": "{}/{}/generated/model".format(base_url, wiki_subfolder),
}

model = EntityDisambiguation(base_url, wiki_subfolder, config, reset_embeddings=False)
predictions, timing = model.predict(mentions_dataset)
```

Optionally users may want to process the results in a predefined format of 
`(start_pos, length, entity, confidence_md, confidence_ed)` per entity found in a given document.

```python
result = process_results(mentions_dataset, predictions, input, include_conf=True)
```

## Replacing the Mention Detection module
With this project we attempt to advocate a modular approach to development, making it easy
for a user to replace certain components. One of such components is the Mention Detection module. After experimenting with
various approaches, we came to the conclusion that the NER-system provided by Flair worked best and was easiest to integrate
in our existing pipeline. We, however, made it easy for the user to replace this part of the pipeline. Luckily, replacing
the module is quite straightforward as we will illustrate by example. Given your own mention detection system, the system
will produce, given a text, a set of spans which have a start position and length. For our API this simply means
replacing the NER-tagger as shown below.

```python
def produce_spans(text):
    return [(0, 5), (17, 7), (50, 6)]
    

tagger_ner = produce_spans
```

For our pipeline integration this means integrating the respective function in the preprocessing step. 

```python
def example_preprocessing():
    # user does some stuff, which results in the format below.
    text = "Obama will visit Germany. And have a meeting with Merkel tomorrow."
    processed = {"test_doc1": [text, produce_spans(text)], 
    "test_doc2": [text, produce_spans(text)}
    return processed
    
input = example_preprocessing()
```

Once the data is in the format above, we still need to parse it such that it may be used for our EL step. We developed
a function for this called `format_spans`. This will return the formatted spans that users provide and can then be used
to make predictions using the ED step. In this case, the contribution of our package will solely be Entity Disambiguation.

```python
mention_detection = MentionDetection(base_url, wiki_subfolder)
mentions_dataset, n_mentions = mention_detection.format_spans(input)
```