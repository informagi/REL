# REL: Radboud Entity Linker
REL is a modular Entity Linking package that can both be integrated in existing pipelines or be used as an API. REL has various meanings, one might first notice that it stands for relation, which is a suiting name for
the problems that can be tackled with this package. Additionally, in Dutch a 'rel' means a disturbance of the public order, which is exactly what we aim to achieve with the release of this package.

# Results

# Setup
## Installation
Please run the following command in a terminal to install REL:
```
pip install git+https://github.com/mickvanhulst/REL
```

## Download
The files used for this project can be divided into three categories. The first is a generic set of documents and embeddings that was used throughout
the project. This folder includes the GloVe embeddings used by Le et al. and the unprocessed datasets that were used to train
the ED model. The second and third category are Wikipedia corpus related files, which in our case either originate from a 2014 or 
2019 corpus. Alternatively, users may use their own corpus, for which we refer to the tutorials.

Download generic
Download 2014
Download 2019.

## Tutorials
To promote usage of this package we developed several tutorials. If you feel one is missing or unclear, then please
create an issue, which is much appreciated :)! The first two tutorials are
for users who simply want to use our package for EL/ED and will be using the data files that we provide. 
The remainder of the tutorials are optional and for users who wish to e.g. train their own Embeddings.

1. [How to get started (project folder and structure).](https://github.com/mickvanhulst/REL/tree/master/tutorials/01_How_to_get_started.md)
2. [End-to-End Entity Linking.](https://github.com/mickvanhulst/REL/tree/master/tutorials/02_E2E_Entity_Linking.md)
3. [(optional) Evaluate on GERBIL.](https://github.com/mickvanhulst/REL/tree/master/tutorials/03_Evaluate_Gerbil.md)
4. [(optional) Extracting a new Wikipedia corpus and creating a p(e|m) index.](https://github.com/mickvanhulst/REL/tree/master/tutorials/04_Extracting_a_new_Wikipedia_corpus.md)
5. [(optional) Training your own Embeddings.](https://github.com/mickvanhulst/REL/tree/master/tutorials/05_training_your_own_embeddings.md)
6. [(optional) Generating training, validation and test files.](https://github.com/mickvanhulst/REL/tree/master/tutorials/06_generating_training_test_files.md)
7. [(optional) Training your own Entity Disambiguation model.](https://github.com/mickvanhulst/REL/tree/master/tutorials/07_training_your_own_ED_model.md)

# Cite
How to cite us.

# Contact
Please email your questions or comments to [Mick van Hulst]('mick.vanhulst@gmail.com)

# Acknowledgements
Our thanks go out to many authors that open-sourced their code, enabling us to this package that can hopefully be of service to many.

In specific, we would like to thank ZalandoResearch for their  Flair package, Le et al. for open-sourcing their
ED code, Kolitsas et al. for providing trained embeddings and a framework that could be used to connect with the GERBIL API, 
Ganea et al. for developing the code required to train the embeddings that were used for the Wikipedia 2014 corpus, and finally the team
behind Wikipedia2Vec that made it possible to develop embeddings for new Wikipedia corpuses.

# License 
