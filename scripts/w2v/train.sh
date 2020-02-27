#! /bin/bash
# Install package again
pip install wikipedia2vec --user

# Uses the previously created files to train the wikipedia2vec model
wikipedia2vec train-embedding dump_file dump_dict wikipedia2vec_trained --link-graph dump_graph --mention-db dump_mention  --dim-size 500 

# Stores the model in required format.
wikipedia2vec save-text --out-format word2vec wikipedia2vec_trained wikipedia2vec_wv2vformat
