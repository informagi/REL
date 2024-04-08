# Mention Detection models

REL offers different named entity models for mention detection:

- `flair`: named model for English, expects upper and lower case text (default)
- `bert_base_cased`: basic name model for English, expects upper and lower case text
- `bert_base_uncased`: basic name model for English, expects lower case text
- `bert_large_cased`: extensive name model for English, expects upper and lower case text
- `bert_large_uncased`: extensive name model for English, expects lower case text
- `bert_multilingual`: multilingual name model, expects upper and lower case text

To change the default Flair model, specify the required model with the `--tagger_ner_name` option, for example when calling the server:

```bash
python src/REL/server.py --tagger_ner_name bert_base_cased
```

or specify the model in the `tagger_name` parameter of a mention detection call:

```python
mentions_dataset, n_mentions = mention_detection.find_mentions(docs, tagger_ner="bert_base_cased")
```

The available named entity models are specified in the file `src/REL/ner/set_tagger_ner.py`. The file names refer to locations on the website huggingface.co, for example https://huggingface.co/flair/ner-english-fast . The file can be extended with new models, for example for other languages.
