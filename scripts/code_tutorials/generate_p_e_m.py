from REL.wikipedia import Wikipedia
from REL.wikipedia_yago_freq import WikipediaYagoFreq

base_url = ""
wiki_version = ""

# 1. Import helper functions; store p(e|m) index etc in class.
print("Loading wikipedia files")
wikipedia = Wikipedia(base_url, wiki_version)

# 2. Init class
wiki_yago_freq = WikipediaYagoFreq(base_url, wiki_version, wikipedia)

# 3. compute Wiki and yago, add parameter to compute_yago to replace Yago with custom count.
# Note: Custom should be of the format:
# {mention_0: {entity_0: cnt_0, entity_n: cnt_n}, ... mention_n: {entity_0: cnt_0, entity_n: cnt_n}}
wiki_yago_freq.compute_wiki()
wiki_yago_freq.compute_custom()

# 4. Store dictionary's in sqlite3 database.
wiki_yago_freq.store()
