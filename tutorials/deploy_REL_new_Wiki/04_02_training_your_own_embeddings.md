# Embeddings
## Training Wikipedia2Vec embeddings
Training new embeddings is based on the [Wikipedia2Vec](http://wikipedia2vec.github.io/) package. For extra information
about this package we refer to their website. We did, however, feel obligated to provide users with the same scripts that
we used to train our embeddings. These two shell scripts first install Wikipedia2vec and then asks you where
your Wikipedia dump is stored. Please make sure that the dump is still zipped and thus has the extensions `.xml.bz2`.
The two scripts are located in `scripts/w2v`, where you first run `preprocess.sh` which requires you to enter the
location of your Wikipedia dump. After this is done, you can run `train.sh` which will train a Wikipedia2Vec model and
store it in the required word2vec format.

## Storing Embeddings in DB
Now that the Embeddings are trained and stored you might notice that the file is huge. This is exactly the reason
why we choose for a database approach, because it was simply infeasible to load all the embeddings into memory. After 
the package is installed, all we have to do is run the code below. Please make sure to not not change the variables `save_dir`
and `db_name`. The variable `embedding_file` needs to point to the trained Wikipedia2vec file.

```python
from REL.db.generic import GenericLookup

save_dir = "{}/{}/generated/".format(base_url, wiki_version)
db_name = "entity_word_embedding"
embedding_file = "./enwiki_w2v_model"

# Embedding load.
emb = GenericLookup(db_name, save_dir=save_dir, table_name='embeddings')
emb.load_word2emb(embedding_file, batch_size=5000, reset=True)
```