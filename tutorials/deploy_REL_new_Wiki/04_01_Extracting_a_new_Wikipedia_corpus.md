# Creating a folder structure
Previously we have defined our `base_url`, but now we need to create a new sub folder in our directory to obtain 
the following folder structure:

```
.
├── generic
└─── wiki_2014
|   ├── basic_data
|      └── anchor_files
|   └── generated
└─── wiki_2019
|   ├── basic_data
|      └── anchor_files
|   └── generated
└─── your_corpus_name
|   ├── basic_data
|      └── anchor_files
|   └── generated
```
# Extracting a Wikipedia dump
There are several platforms that host [Wikipedia dumps](https://dumps.wikimedia.org/). These platforms provide `xml` files that need processing. 
A tool that does this is called [WikiExtractor](https://github.com/attardi/wikiextractor). This tool takes as an input a
Wikipedia dump and spits out files that are required for our package. We, however, had to alter it slightly such that it 
stored some additional files that are required for this package. As such, we have added this edited edition to our scripts
folder. To process a Wikipedia dump run the command below in a terminal. We define the file size (`bytes`) as one GB, but it can
be changed based on the user's wishes. We advice users to run the script in the `basic_data` folder. After the script is
done, the user only needs to copy the respective wikipedia dump (this excludes the wiki id/name mapping, disambiguation pages
and redirects) into the `anchor_files` folder. 

```
python WikiExtractor.py ./wiki_corpus.xml --links --filter_disambig_pages --processes 1 --bytes 1G
```

# Generate p(e|m) index
Now that we have extracted the necessary data from our Wikipedia corpus, we may create the p(e|m) index. This index
is automatically stored in the same database as the embeddings that can be found in the `generated` folder. The first
thing we need to do is define define a variable where we store our database. Secondly, we instantiate a `Wikipedia` class
that loads the wikipedia id/name mapping, disambiguation file and redirect file. 

```python
wiki_version = "your_corpus_name"
wikipedia = Wikipedia(base_url, wiki_version)
```

Now all that is left is to instantiate our `WikipediaYagoFreq` class that is responsible for parsing the Wikipedia and
YAGO articles. Here we note that the function `compute_custom()` by default computes the p(e|m) probabilities of
YAGO, but that it can be replaced by any Knowledge Base of your choosing. To replace YAGO, make sure that the input dictionary
to the aforementioned function is of the following format: 
`{mention_0: {entity_0: cnt_0, entity_n: cnt_n}, ... mention_n: {entity_0: cnt_0, entity_n: cnt_n}}`

```python
wiki_yago_freq = WikipediaYagoFreq(base_url, wiki_version, wikipedia)
wiki_yago_freq.compute_wiki()
wiki_yago_freq.compute_custom()
wiki_yago_freq.store()
```
