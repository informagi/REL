# Deploy REL for a new Wikipedia corpus
Although we will do our best to continuously provide the community with recent corpuses, it may be the case that a user wants to, for example, use
an older corpus for a specific evaluation. For this reason we provide the user with the option to do so. We must, however,
note that some steps are outside the scope of the REL package, which makes support for some of these steps a difficult task.

This tutorial is divided in four parts. The first part deals with [Extracting a Wikipedia corpus and creating a p(e|m) index](https://github.com/informagi/REL/tree/master/tutorials/deploy_REL_new_Wiki/04_01_Extracting_a_new_Wikipedia_corpus.md).
After extracting the aforementioned index and thus obtaining a sqlite3 database, we are also in need of Embeddings. We obtain new embeddings by [training a Wikipedia2Vec model](https://github.com/informagi/REL/tree/master/tutorials/deploy_REL_new_Wiki/04_02_training_your_own_embeddings.md).
To train our own Entity Disambiguation model, we need to [generate training, validation and test files](https://github.com/informagi/REL/tree/master/tutorials/deploy_REL_new_Wiki/04_03_generating_training_test_files.md).
These aforementioned p(e|m) index,  can be used to [train your own Entity Disambiguation model](https://github.com/informagi/REL/tree/master/tutorials/deploy_REL_new_Wiki/04_04_training_your_own_ED_model.md). 
After obtaining this model, a user may choose to [evaluate the obtained model on Gerbil](https://github.com/informagi/REL/tree/master/tutorials/03_Evaluate_Gerbil.md) or for [E2E Entity Linking](https://github.com/informagi/REL/tree/master/tutorials/02_E2E_Entity_Linking.md).