from flair.models import SequenceTagger

from REL.entity_disambiguation import EntityDisambiguation
from REL.mention_detection import MentionDetection
from REL.ner.ngram import Cmns
from REL.utils import process_results


def example_preprocessing():
    # Example splitting, should be of format {doc_1: {sent_idx: [sentence, []]}, .... }}
    text = """Obama will visit Germany. And have a meeting with Merkel tomorrow.
    Obama will visit Germany. And have a meeting with Merkel tomorrow. Go all the way or blah blah Charles Bukowski."""
    spans = []  # [(0, 5), (17, 7), (50, 6)]
    processed = {"test_doc": [text, spans], "test_doc2": [text, spans]}
    return processed


base_url = "/users/vanhulsm/Desktop/projects/data/"
wiki_version = "wiki_2014"

# 1. Input sentences when using Flair.
input = example_preprocessing()

# For Mention detection two options.
# 2. Mention detection, we used the NER tagger, user can also use his/her own mention detection module.
mention_detection = MentionDetection(base_url, wiki_version)

# 2. Alternatively. if you want to use your own MD system (or ngram detection),
# the required input is: {doc_name: [text, spans] ... }.
mentions_dataset, n_mentions = mention_detection.format_spans(input)

# 2. Alternative MD module is using an n-gram tagger.
<<<<<<< HEAD
# tagger_ner = SequenceTagger.load("ner-fast")
tagger_ngram = Cmns(base_url, wiki_subfolder, n=5)
=======
#tagger_ner = SequenceTagger.load("ner-fast")
tagger_ngram = Cmns(base_url, wiki_version, n=5)
>>>>>>> master

mentions_dataset, n_mentions = mention_detection.find_mentions(input, tagger_ngram)

# 3. Load model.
config = {
    "mode": "eval",
<<<<<<< HEAD
    "model_path": "{}/{}/generated/model".format(base_url, wiki_subfolder),
=======
    "model_path": "{}/{}/generated/model".format(
        base_url, wiki_version
    ),
>>>>>>> master
}
model = EntityDisambiguation(base_url, wiki_version, config)

# 4. Entity disambiguation.
predictions, timing = model.predict(mentions_dataset)

# 5. Optionally use our function to get results in a usable format.
result = process_results(mentions_dataset, predictions, input)

print(result)
