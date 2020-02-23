# Generating training, validation and test files
To train your own Entity Disambiguation model, training, validation and test files are required. To obtain these
files we first define our `wiki_version` and instantiate a `Wikipedia` class that are required.
```python
from REL.wikipedia import Wikipedia
from REL.generate_train_test import GenTrainingTest

wiki_version = "wiki_2019/"
wikipedia = Wikipedia(base_url, wiki_version)
```
Secondly, we instantiate the class `GenTrainingTest` that parses the raw training and test files that can be found in our
`generic` folder. The user may choose to only retrieve one of the listed datasets below by simply changing the name
in the function `process_wned` or `process_aida`. The reason for separating these functions was due to the way they were
provided and had to be parsed.

```python
data_handler = GenTrainingTest(base_url, wiki_version, wikipedia)
for ds in ["aquaint", "msnbc", "ace2004", "wikipedia", "clueweb"]:
    data_handler.process_wned(ds)

for ds in ["train", "test"]:
    data_handler.process_aida(ds)
```