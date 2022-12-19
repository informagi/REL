import json
import logging
import sqlite3
from array import array
from functools import lru_cache
from os import makedirs, path
import numpy as np

import requests


class DB:
    @staticmethod
    def download_file(url, local_filename):
        """
        Downloads a file from an url to a local file.
        Args:
            url (str): url to download from.
            local_filename (str): local file to download to.
        Returns:
            str: file name of the downloaded file.
        """
        r = requests.get(url, stream=True, verify=False)
        if path.dirname(local_filename) and not path.isdir(
            path.dirname(local_filename)
        ):
            raise Exception(local_filename)
            makedirs(path.dirname(local_filename))
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return local_filename

    def initialize_db(self, fname, table_name, columns):
        """
        Args:
            fname (str): location of the database.
        Returns:
            db (sqlite3.Connection): a SQLite3 database with an embeddings table.
        """
        # open database in autocommit mode by setting isolation_level to None.
        db = sqlite3.connect(fname, isolation_level=None, check_same_thread=False)

        q = "create table if not exists {}(word text primary key, {})".format(
            table_name, ", ".join(["{} {}".format(k, v) for k, v in columns.items()])
        )
        db.cursor().execute(q)
        return db

    def create_index(self, columns=None, table_name=None):
        # if columns:
        #     self.columns = columns
        #     self.table_name = table_name
        #

        # for i, (k, v) in enumerate(self.columns.items()):
        #     createSecondaryIndex = "CREATE INDEX if not exists idx_{} ON {}({})".format(
        #         k, self.table_name, k
        #     )
        #     print(createSecondaryIndex)
        #     c.execute(createSecondaryIndex)
        createSecondaryIndex = "CREATE INDEX if not exists idx_{} ON {}({})".format(
            "lower", "wiki", "lower"
        )
        print(createSecondaryIndex)
        self.cursor.execute(createSecondaryIndex)

    def clear(self):
        """
        Deletes all embeddings from the database.
        """
        self.cursor.execute("delete from {}".format(self.table_name))

    def insert_batch_emb(self, batch):
        """
        Args:
            batch (list): a list of embeddings to insert, each of which is a tuple ``(word, embeddings)``.
        Example:
        .. code-block:: python
            e = Embedding()
            e.db = e.initialize_db(self.e.path('mydb.db'))
            e.insert_batch([
                ('hello', [1, 2, 3]),
                ('world', [2, 3, 4]),
                ('!', [3, 4, 5]),
            ])
        """
        binarized = [(word, array("f", emb).tobytes()) for word, emb in batch]
        try:
            # Adding the transaction statement reduces total time from approx 37h to 1.3h.
            self.cursor.execute("BEGIN TRANSACTION;")
            self.cursor.executemany(
                "insert into {} values (?, ?)".format(self.table_name), binarized
            )
            self.cursor.execute("COMMIT;")
        except Exception as e:
            print("insert failed\n{}".format([w for w, e in batch]))
            raise e

    def insert_batch_wiki(self, batch):
        """
        Args:
            batch (list): a list of embeddings to insert, each of which is a tuple ``(word, embeddings)``.
        Example:
        .. code-block:: python
            e = Embedding()
            e.db = e.initialize_db(self.e.path('mydb.db'))
            e.insert_batch([
                ('hello', [1, 2, 3]),
                ('world', [2, 3, 4]),
                ('!', [3, 4, 5]),
            ])
        """
        binarized = [
            (word, json.dumps(p_e_m).encode(), lower, occ)
            for word, p_e_m, lower, occ in batch
        ]
        try:
            # Adding the transaction statement reduces total time from approx 37h to 1.3h.
            self.cursor.execute("BEGIN TRANSACTION;")
            self.cursor.executemany(
                "insert into {} values (?, ?, ?, ?)".format(self.table_name), binarized
            )
            self.cursor.execute("COMMIT;")
        except Exception as e:
            print("insert failed\n{}".format([w for w, e in batch]))
            raise e

    def lookup_list(self, w, table_name, column="emb"):
        """
        Args:
            w: list of words to look up.
        Returns:
            embeddings for ``w``, if it exists.
            ``None``, otherwise.
        """
        w = list(w)

        if len(w) == 0:
            res = []
        elif len(w) == 1:
            e = self.lookup(column, table_name, w[0])
            res = [e if e is None else np.frombuffer(e[0], dtype=np.float32)]
        else:
            ret = self.lookup_many(column, table_name, w)
            mapping = {key: np.frombuffer(value, dtype=np.float32) for key, value in ret}
            res = [mapping.get(word) for word in w]

        return res

    def lookup_many(self, column, table_name, w):
        qmarks = ','.join(('?',)*len(w))
        return self.cursor.execute(
            f"select word,{column} from {table_name} where word in ({qmarks})",
            w,
        ).fetchall()

    @lru_cache(maxsize=None)
    def lookup(self, column, table_name, word):
        return self.cursor.execute(
            f"select {column} from {table_name} where word = :word",
            {"word": word},
        ).fetchone()

    @lru_cache(maxsize=None)
    def lookup_wik(self, w, table_name, column):
        """
        Args:
            w: word to look up.
        Returns:
            embeddings for ``w``, if it exists.
            ``None``, otherwise.
        """
        # q = c.execute('select emb from embeddings where word = :word', {'word': w}).fetchone()
        # return array('f', q[0]).tolist() if q else None
        if column == "lower":
            e = self.cursor.execute(
                "select word from {} where {} = :word".format(table_name, column),
                {"word": w},
            ).fetchone()
        else:
            e = self.cursor.execute(
                "select {} from {} where word = :word".format(column, table_name),
                {"word": w},
            ).fetchone()
        res = (
            e if e is None else json.loads(e[0].decode()) if column == "p_e_m" else e[0]
        )

        return res

    def ensure_file(self, name, url=None, logger=logging.getLogger()):
        """
        Ensures that the file requested exists in the cache, downloading it if it does not exist.
        Args:
            name (str): name of the file.
            url (str): url to download the file from, if it doesn't exist.
            force (bool): whether to force the download, regardless of the existence of the file.
            logger (logging.Logger): logger to log results.
            postprocess (function): a function that, if given, will be applied after the file is downloaded. The function has the signature ``f(fname)``
        Returns:
            str: file name of the downloaded file.
        """
        fname = "{}/{}".format(self.save_dir, name)
        if not path.isfile(fname):
            if url:
                logger.critical("Downloading from {} to {}".format(url, fname))
                DB.download_file(url, fname)
            else:
                raise Exception("{} does not exist!".format(fname))
        return fname
