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
from REL.ngram import Cmns

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

model = EntityDisambiguation(base_url, wiki_subfolder, config)
```

As was mentioned prior to this, we used Flair's NER tagger as our Mention Detection system. This system can be replaced 
with either our n-gram system or your own custom MD module. Using your own MD module will be elaborated on below. 
For now we assume that you want to either use Flair's NER tagger or our n-gram detection. For high Recall tasks we advice 
the use of our n-gram module. Note that the parameter `n` refers to the max to-be-considered length of a candidate mention.

```python
tagger_ner = SequenceTagger.load("ner-fast")
tagger_ngram = Cmns(base_url, wiki_subfolder, n=5)
```

Our final step consists of starting the server, where we may define our IP-address and port. Additionally,
the user may choose to include or exclude confidence scores.

```python
server_address = ("127.0.0.1", 1235)
server = HTTPServer(
    server_address,
    make_handler(
        base_url, wiki_subfolder, model, tagger_ngram
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
server with our package and query it with multiple people. A user may query the API using the code below, where the `text_doc` 
variable may be replaced with a text for a particular document. At this point in time, a user may only query the server with
a single document. We currently added this to make sure that a single user does not overload a particular server. Additionally,
if a user wishes to predict in an ED-fashion only, then the spans key should not be left empty and should be filled with tuples
with integer values `(start_pos, mention_length)` (starting position and length of mention respectively).

```python
import requests

IP_ADDRESS = "http://localhost"
PORT = "1235"
text_doc = "If you're going to try, go all the way - Charles Bukowski"

document = {
    "text": text_doc,
    "spans": [],
}

API_result = requests.post("{}:{}".format(IP_ADDRESS, PORT), json=document).json()
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
from REL.ngram import Cmns

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

Now that we have defined our `input` we instantiate our mention detection class and MD module, and then use both to find
mentions of our dataset. The output of the function `find_mentions()` is a dictionary with various properties that
are required for the Entity Disambiguation module. Additionally, it returns a count consisting of the total number of
mentions that were found using the NER-tagger. This number may be unequal to the number of mentions that the dictionary
`mentions_dataset` stores as this dictionary only stores mentions that were found in our Knowledge Base.

```python
mention_detection = MentionDetection(base_url, wiki_subfolder)
tagger_ner = SequenceTagger.load("ner-fast")
tagger_ngram = Cmns(base_url, wiki_subfolder, n=5)
mentions_dataset, n_mentions = mention_detection.find_mentions(input, tagger_ngram)
```

The final step of our End-to-End process consists of instantiating the model and using it to find entities based
on the mentions that we previously found.

```python
config = {
    "mode": "eval",
    "model_path": "{}/{}/generated/model".format(base_url, wiki_subfolder),
}

model = EntityDisambiguation(base_url, wiki_subfolder, config)
predictions, timing = model.predict(mentions_dataset)
```

Optionally users may want to process the results in a predefined format of 
`(start_pos, length, entity, NER-type, confidence_md, confidence_ed)` per entity found in a given document.

```python
result = process_results(mentions_dataset, predictions, input)
```

## Replacing the Mention Detection module
With this project we attempt to advocate a modular approach to development, making it easy
for a user to replace certain components. One of such components is the Mention Detection module. After experimenting with
various approaches, we came to the conclusion that the NER-system provided by Flair worked best and was easiest to integrate
in our existing pipeline. We, however, made it easy for the user to replace this part of the pipeline. Luckily, replacing
the module is quite straightforward as we will illustrate by example. We have defined an example MD class below,
which given a particular sentence and all of the sentences of that given text (for global context), produces a set
of mentions for that sentence.

```python
class Token:
    def __init__(self, text, start_pos, end_pos, score, tag):
        self.text = text
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.score = score
        self.tag = tag

class MD_Module(object):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def predict(self, sentence, sentences_doc):
        """
        The module takes as an input a sentence from the current doc and all the sentences of
        the current doc (for global context). The user is expected to return a list of mentions,
        where each mention is a Token class.

        We denote the following requirements:
        1. Any MD module should have a 'predict()' function that returns a list of mentions.
        2. A mention is always defined as a Token class (see above).

        """
        # returns list of Token objects.
        return self.find_mentions()

    def find_mentions(self, sentence):
        mentions = []
        for i in range(10):
            mentions.append(Token(i, i, i, i, i))
        return mentions
```

For our pipeline integration this means replcacing either `tagger_ner` or `tagger_ngram` with the class definition below.

```python
tagger_custom = MD_Module('param1', 'param2')
```