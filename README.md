# REL: Radboud Entity Linker
REL is a modular Entity Linking package that can both be integrated in existing pipelines or be used as an API. REL has various meanings, one might first notice that it stands for relation, which is a suiting name for
the problems that can be tackled with this package. Additionally, in Dutch a 'rel' means a disturbance of the public order, which is exactly what we aim to achieve with the release of this package.

# Setup API
This section elaborates on how a user may utilise our API. Steps include obtaining
a API key and querying our API. Please note that at this point in time we do not require obtaining
a key and leave it for future work.

## Obtaining a key
Not necessary at this point in time, please continue to the next step.

## Querying our API
Users may access our API by using the example script below. For EL, the user should leave the `spans` field empty. Additionally,
if a user wishes to predict in an ED-fashion only, then the spans key should not be left empty and should be filled with tuples
consisting of integer values that represent the starting position and length of the mention respectively.

```python
import requests

IP_ADDRESS = "http://gem.cs.ru.nl/api"
PORT = "80"
text_doc = "If you're going to try, go all the way - Charles Bukowski"

# Example EL.
document = {
    "text": text_doc,
    "spans": []
}

# Example ED.
document = {
    "text": text_doc,
    "spans": [(41, 16)]
}


API_result = requests.post("{}:{}".format(IP_ADDRESS, PORT), json=document).json()
```

# Setup package
The following installation, downloads and installation focuses on the local-usage of our package. If a user wishes
to use our API, then we refer to the section above.

## Installation
Please run the following command in a terminal to install REL:
```
pip install git+https://github.com/informagi/REL
```

## Download
The files used for this project can be divided into three categories. The first is a generic set of documents and embeddings that was used throughout
the project. This folder includes the GloVe embeddings used by Le et al. and the unprocessed datasets that were used to train
the ED model. The second and third category are Wikipedia corpus related files, which in our case either originate from a 2014 or 
2019 corpus. Alternatively, users may use their own corpus, for which we refer to the tutorials.

[Download generic files](https://drive.google.com/file/d/15rz-q7ohCIZg-2hVqYojaos4ag79phSd/view?usp=sharing)

[Download Wikipedia corpus (2014)](https://drive.google.com/file/d/1BhUA7h6PaP7ZcFJpLZzXxAH3k_EK1Iw-/view?usp=sharing)

[Download Wikipedia corpus (2019)](https://drive.google.com/file/d/1Baxh36Eg0zhZ60PFRL4bjP8Z7Tz8F4hk/view?usp=sharing)

## Tutorials
To promote usage of this package we developed various tutorials. If you simply want to use our API, then 
we refer to the section above. If you feel one is missing or unclear, then please create an issue, which is much appreciated :)! The first two tutorials are
for users who simply want to use our package for EL/ED and will be using the data files that we provide. 
The remainder of the tutorials are optional and for users who wish to e.g. train their own Embeddings.

1. [How to get started (project folder and structure).](https://github.com/informagi/REL/tree/master/tutorials/01_How_to_get_started.md)
2. [End-to-End Entity Linking.](https://github.com/informagi/REL/tree/master/tutorials/02_E2E_Entity_Linking.md)
3. [Evaluate on GERBIL.](https://github.com/informagi/REL/tree/master/tutorials/03_Evaluate_Gerbil.md)
4. [Deploy REL for a new Wikipedia corpus](https://github.com/informagi/REL/tree/master/tutorials/deploy_REL_new_Wiki/04_deploy_REL_new_wiki.md):
    1. [Extracting a new Wikipedia corpus and creating a p(e|m) index.](https://github.com/informagi/REL/tree/master/tutorials/deploy_REL_new_Wiki/04_01_Extracting_a_new_Wikipedia_corpus.md)
    2. [Training your own Embeddings.](https://github.com/informagi/REL/tree/master/tutorials/deploy_REL_new_Wiki/04_02_training_your_own_embeddings.md)
    3. [Generating training, validation and test files.](https://github.com/informagi/REL/tree/master/tutorials/deploy_REL_new_Wiki/04_03_generating_training_test_files.md)
    4. [Training your own Entity Disambiguation model.](https://github.com/informagi/REL/tree/master/tutorials/deploy_REL_new_Wiki/04_04_training_your_own_ED_model.md)
5. [Reproducing our results](https://github.com/informagi/REL/tree/master/tutorials/05_reproducing_our_results.md)
6. [REL as systemd service](https://github.com/informagi/REL/tree/master/tutorials/06_systemd_instructions.md)

# Cite
How to cite us.

# Contact
Please email your questions or comments to [Mick van Hulst]('mick.vanhulst@gmail.com)

# Acknowledgements
Our thanks go out to the authors that open-sourced their code, enabling us to this package that can hopefully be of service to many.
