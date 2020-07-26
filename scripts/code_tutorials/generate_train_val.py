from REL.generate_train_test import GenTrainingTest
from REL.wikipedia import Wikipedia

base_url = "/Users/vanhulsm/Desktop/projects/data/"
wiki_version = "wiki_2014"
wikipedia = Wikipedia(base_url, wiki_version)

data_handler = GenTrainingTest(base_url, wiki_version, wikipedia)

for ds in ["test"]:
    data_handler.process_aida(ds)

for ds in ["aquaint"]:
    data_handler.process_wned(ds)

for ds in ['gerdaq_devel', 'gerdaq_test', 'gerdaq_train']:
    data_handler.process_gerdaq(ds)