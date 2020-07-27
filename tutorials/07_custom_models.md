# Notes on using custom models with REL
It is easy to swap out parts of REL's pipeline with different models. REL
contains infrastructure to easily download and use models provided by us, but
you can also define your own models. In this tutorial, we outline some possible
scenarios.

Everything listed below only works for *loading* models. When training a new
model, you can only use a local filepath.

- [Notes on using custom models with REL](#notes-on-using-custom-models-with-rel)
- [Loading semantics](#loading-semantics)
  - [Examples](#examples)
    - [Aliases](#aliases)
    - [Local filepath](#local-filepath)
    - [Local or remote archives](#local-or-remote-archives)
- [Under the hood](#under-the-hood)

# Loading semantics
NER and ED models that we provide as part of REL can be loaded easily using
aliases.  Available models are listed
[here](https://github.com/informagi/REL/tree/master/REL/models/models.json).
All models that need to be downloaded from the web are cached for subsequent
use.

*TODO: list evaluations of models here, a la Flair*

Additionally, fields that accept aliases also take URLs and regular file paths
as values.

## Examples
Loading Flair NER models:
```python
from REL.ner import load_flair_ner

### Flair ###
# Flair provides their own aliases, you can use those as well as our aliases.
# For example, using ner-fast from Flair:
ner_tagger = load_flair_ner("ner-fast")

# Or using ner-fast-with-lowercase, a finetuned version of ner-fast provided
# as part of REL:
ner_tagger = load_flair_ner("ner-fast-with-lowercase")

# You can also use a URL (http or https) to a model checkpoint, e.g.:
ner_tagger = load_flair_ner("https://some.website.com/flair_model.pt")

# Or a path on your filesystem:
ner_tagger = load_flair_ner("/home/user/flair/model.pt")
```

---

You can use the same variations when loading `EntityDisambiguation` models, but
there are a few caveats. ED models are stored in two files, `model.state_dict`
(the actual weights) and `model.config` (architecture parameters etc.). To load
the model, you can supply either a local filepath without extension, or a
URL/filepath to a tarfile containing the two files. Examples of the required
directory structure are shown below.

### Aliases
Loading an ED model using a REL alias is simple. Pass the alias string to the
`model_path` key in the configuration, e.g.:
```python
from REL.entity_disambiguation import EntityDisambiguation

base_url = "/path/to/some/place/"
wiki_version = "wiki_2019"
config = {
    "mode": "eval",
    "model_path": "ed-wiki-2019"  # model alias
}
ed_model = EntityDisambiguation(base_url, wiki_version, config)
```

### Local filepath
Required directory structure:
```
model_folder
├── model.config
└── model.state_dict
```
Note that the files *must* be named `model.state_dict` and `model.config`!  If
you wish to use a logistic regression model for confidence estimation, its
checkpoint must be called `lr_model.pkl` and be in the same directory as the
model files.

Loading this model:
```python
from REL.entity_disambiguation import EntityDisambiguation

base_url = "/path/to/some/place/"
wiki_version = "wiki_2019"
config = {
    "mode": "eval",
    "model_path": "model_folder/model"  # partial filepath to model
}
ed_model = EntityDisambiguation(base_url, wiki_version, config)
```

### Local or remote archives
Required archive structure:
```
Filename: model_folder.tar
Contents:

model_folder
├── model.config
└── model.state_dict
```
The archive filename (minus extension) should be identical to the folder it
contains.  The model files again *must* be called `model.state_dict` and
`model.config` (and `lr_model.pkl` if applicable). Compressed archives are also
supported, see Python's
[tarfile](https://docs.python.org/3/library/tarfile.html) module documentation
for supported formats.

Loading a model from local archive:
```python
from REL.entity_disambiguation import EntityDisambiguation

base_url = "/path/to/some/place/"
wiki_version = "wiki_2019"
config = {
    "mode": "eval",
    "model_path": "/path/to/model_folder.tar"  # filepath to archive
}
ed_model = EntityDisambiguation(base_url, wiki_version, config)
```

Loading a model from remote archive:
```python
from REL.entity_disambiguation import EntityDisambiguation

base_url = "/path/to/some/place/"
wiki_version = "wiki_2019"
config = {
    "mode": "eval",
    "model_path": "http://some.site.com/model_folder.tar"  # URL to archive
}
ed_model = EntityDisambiguation(base_url, wiki_version, config)
```

# Under the hood
All the remote model loading functionality is built on `REL.utils.fetch_model`.
This function tries to get a URL from `models.json` if applicable, and then
downloads and caches the model checkpoint. You can use this function yourself
to download and cache any other files you might need as well, it is not
inherently limited to just model checkpoints.
