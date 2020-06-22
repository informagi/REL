from REL.wikipedia import Wikipedia
from REL.generate_train_test import GenTrainingTest

base_url = "/Users/vanhulsm/Desktop/projects/data/"
wiki_version = "wiki_2014"
wikipedia = Wikipedia(base_url, wiki_version)

data_handler = GenTrainingTest(base_url, wiki_version, wikipedia)

<<<<<<< HEAD
# for ds in ["aquaint", "msnbc", "ace2004", "wikipedia", "clueweb"]:
#     data_handler.process_wned(ds)

for ds in ["test"]:  # , "train"]:
=======
for ds in ["test"]:
>>>>>>> master
    data_handler.process_aida(ds)

for ds in ["aquaint"]:
    data_handler.process_wned(ds)
