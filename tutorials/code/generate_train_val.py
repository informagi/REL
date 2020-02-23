from REL.wikipedia import Wikipedia
from REL.generate_train_test import GenTrainingTest

base_url = "C:/Users/mickv/Google Drive/projects/entity_tagging/deep-ed/data/"
wiki_version = "wiki_2019/"
wikipedia = Wikipedia(base_url, wiki_version)

data_handler = GenTrainingTest(base_url, wiki_version, wikipedia)

for ds in ["aquaint", "msnbc", "ace2004", "wikipedia", "clueweb"]:
    data_handler.process_wned(ds)

for ds in ["train", "test"]:
    data_handler.process_aida(ds)
