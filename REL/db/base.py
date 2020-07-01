import json
import logging
import sqlite3
from array import array
from os import makedirs, path

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
        db = sqlite3.connect(fname, isolation_level=None)
        c = db.cursor()

        q = "create table if not exists {}(word text primary key, {})".format(
            table_name, ", ".join(["{} {}".format(k, v) for k, v in columns.items()])
        )
        c.execute(q)
        return db

    def create_index(self, columns=None, table_name=None):
        # if columns:
        #     self.columns = columns
        #     self.table_name = table_name
        #
        c = self.db.cursor()
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
        c.execute(createSecondaryIndex)

    def clear(self):
        """
        Deletes all embeddings from the database.
        """
        c = self.db.cursor()
        c.execute("delete from {}".format(self.table_name))

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
        c = self.db.cursor()
        binarized = [(word, array("f", emb).tobytes()) for word, emb in batch]
        try:
            # Adding the transaction statement reduces total time from approx 37h to 1.3h.
            c.execute("BEGIN TRANSACTION;")
            c.executemany(
                "insert into {} values (?, ?)".format(self.table_name), binarized
            )
            c.execute("COMMIT;")
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
        c = self.db.cursor()
        binarized = [
            (word, self.dict_to_binary(p_e_m), lower, occ)
            for word, p_e_m, lower, occ in batch
        ]
        try:
            # Adding the transaction statement reduces total time from approx 37h to 1.3h.
            c.execute("BEGIN TRANSACTION;")
            c.executemany(
                "insert into {} values (?, ?, ?, ?)".format(self.table_name), binarized
            )
            c.execute("COMMIT;")
        except Exception as e:
            print("insert failed\n{}".format([w for w, e in batch]))
            raise e

    def dict_to_binary(self, the_dict):
        # credit: https://stackoverflow.com/questions/19232011/convert-dictionary-to-bytes-and-back-again-python
        str = json.dumps(the_dict)
        binary = " ".join(format(ord(letter), "b") for letter in str)
        return binary

    def binary_to_dict(self, the_binary):
        jsn = "".join(chr(int(x, 2)) for x in the_binary.split())
        d = json.loads(jsn)
        return d

    def lookup(self, w, table_name, column="emb"):
        """
        Args:
            w: word to look up.
        Returns:
            embeddings for ``w``, if it exists.
            ``None``, otherwise.
        """
        c = self.db.cursor()

        res = []
        c.execute("BEGIN TRANSACTION;")
        for word in w:
            e = c.execute(
                "select {} from {} where word = :word".format(column, table_name),
                {"word": word},
            ).fetchone()
            res.append(e if e is None else array("f", e[0]).tolist())
        c.execute("COMMIT;")

        return res

    def lookup_wik(self, w, table_name, column):
        """
        Args:
            w: word to look up.
        Returns:
            embeddings for ``w``, if it exists.
            ``None``, otherwise.
        """
        c = self.db.cursor()
        # q = c.execute('select emb from embeddings where word = :word', {'word': w}).fetchone()
        # return array('f', q[0]).tolist() if q else None
        if column == "lower":
            e = c.execute(
                "select word from {} where {} = :word".format(table_name, column),
                {"word": w},
            ).fetchone()
        else:
            e = c.execute(
                "select {} from {} where word = :word".format(column, table_name),
                {"word": w},
            ).fetchone()
        res = (
            e if e is None else self.binary_to_dict(e[0]) if column == "p_e_m" else e[0]
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
