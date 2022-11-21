import os
from functools import lru_cache
from time import time

import numpy as np
from gensim import utils
from numpy import float32 as REAL
from numpy import zeros

from REL.db.base import DB


class GenericLookup(DB):
    def __init__(
        self,
        name,
        save_dir,
        table_name="embeddings",
        d_emb=300,
        columns={"emb": "blob"},
    ):
        """
        Args:
            name: name of the embedding to retrieve.
            d_emb: embedding dimensions.
            show_progress: whether to print progress.
        """
        self.avg_cnt = {
            "word": {"cnt": 0, "sum": zeros(d_emb)},
            "entity": {"cnt": 0, "sum": zeros(d_emb)},
        }

        path_db = os.path.join(save_dir, f"{name}.db")

        self.d_emb = d_emb
        self.name = name
        self.db = self.initialize_db(path_db, table_name, columns)
        self.cursor = self.db.cursor()
        self.table_name = table_name
        self.columns = columns

    def emb(self, words, table_name):
        g = self.lookup_list(words, table_name)
        return g

    def wiki(self, mention, table_name, column_name="p_e_m"):
        g = self.lookup_wik(mention, table_name, column_name)
        return g

    def load_word2emb(self, file_name, batch_size=5000, limit=np.inf, reset=False):
        self.seen = set()
        if reset:
            self.clear()

        batch = []
        start = time()

        # Loop over file.
        with utils.open(file_name, "rb") as fin:
            # Determine size file.
            header = utils.to_unicode(fin.readline(), encoding="utf-8")
            vocab_size, vector_size = (
                int(x) for x in header.split()
            )  # throws for invalid file format
            if limit < vocab_size:
                vocab_size = limit

            for line_no in range(vocab_size):
                line = fin.readline()
                if line == b"":
                    raise EOFError(
                        "unexpected end of input; is count incorrect or file otherwise damaged?"
                    )

                parts = utils.to_unicode(
                    line.rstrip(), encoding="utf-8", errors="strict"
                ).split(" ")

                if len(parts) != vector_size + 1:
                    raise ValueError(
                        "invalid vector on line %s (is this really the text format?)"
                        % line_no
                    )

                word, vec = parts[0], np.array([REAL(x) for x in parts[1:]])

                if word in self.seen:
                    continue

                self.seen.add(word)
                batch.append((word, vec))

                if "ENTITY/" in word:
                    self.avg_cnt["entity"]["cnt"] += 1
                    self.avg_cnt["entity"]["sum"] += vec
                else:
                    self.avg_cnt["word"]["cnt"] += 1
                    self.avg_cnt["word"]["sum"] += vec

                if len(batch) == batch_size:
                    print("Another {}".format(batch_size), line_no, time() - start)
                    start = time()
                    self.insert_batch_emb(batch)
                    batch.clear()

        for x in ["entity", "word"]:
            if self.avg_cnt[x]["cnt"] > 0:
                batch.append(
                    (
                        "#{}/UNK#".format(x.upper()),
                        self.avg_cnt[x]["sum"] / self.avg_cnt[x]["cnt"],
                    )
                )
                print("Added #{}/UNK#".format(x.upper()))

        if batch:
            self.insert_batch_emb(batch)
        # self.create_index()

    def load_wiki(self, p_e_m_index, mention_total_freq, batch_size=5000, reset=False):
        if reset:

            self.clear()

        batch = []
        start = time()

        for i, (ment, p_e_m) in enumerate(p_e_m_index.items()):
            p_e_m = sorted(p_e_m.items(), key=lambda kv: kv[1], reverse=True)
            batch.append((ment, p_e_m, ment.lower(), mention_total_freq[ment]))

            if len(batch) == batch_size:
                print("Another {}".format(batch_size), time() - start)
                start = time()
                self.insert_batch_wiki(batch)
                batch.clear()

        if batch:
            self.insert_batch_wiki(batch)

        self.create_index()


if __name__ == "__main__":
    save_dir = "C:/Users/mickv/Desktop/data_back/wiki_2019/generated"

    # Test data
    # ent_p_e_m_index = {
    #     "Netherlands": {32796504: 1 / 3, 32796504: 2 / 3},
    #     "Netherlands2": {32796504: 1 / 3, 32796504: 2 / 3},
    # }
    # mention_total_freq = {"Netherlands": 10, "Netherlands2": 100}
    #
    # # Wiki load.
    # wiki = GenericLookup('entity_word_embedding', save_dir=save_dir, table_name='wiki',
    #                      columns={"p_e_m": "blob", "lower": "text", "freq": "INTEGER"})
    # wiki.load_wiki(ent_p_e_m_index, mention_total_freq, reset=True)
    #
    # # Query
    # p_e_m = wiki.wiki("Netherlands", "wiki")
    # freq = wiki.wiki("Netherlands", "wiki", "freq")
    # lowercase = wiki.wiki("Netherlands".lower(), "wiki", "lower")

    # Embedding load.
    emb = GenericLookup(
        "entity_word_embedding", save_dir=save_dir, table_name="embeddings"
    )
    emb.load_word2emb(
        "D:/enwiki-20190701-model-w2v-dim300", batch_size=5000, reset=True
    )

    # Query
    import torch

    embeddings = torch.stack(
        [torch.tensor(e) for e in emb.emb(["in", "the", "end"], "embeddings")]
    )
