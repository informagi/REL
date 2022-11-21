import os
from urllib.parse import unquote

from REL.utils import first_letter_to_uppercase, trim1



class Wikipedia:
    """
    Class responsible for loading Wikipedia files. Required when filling sqlite3 database with e.g. p(e|m) index.
    """
    def __init__(self, base_url, wiki_version):
        self.base_url = base_url + wiki_version
        # if include_wiki_id_name:
        self.wiki_disambiguation_index = self.generate_wiki_disambiguation_map()
        print("Loaded wiki disambiguation index")
        (
            self.wiki_redirects_index,
            self.wiki_redirects_id_index,
        ) = self.generate_wiki_redirect_map()
        print("Loaded wiki redirects index")
        self.wiki_id_name_map = self.gen_wiki_name_map()
        print("Loaded entity index")

    def preprocess_ent_name(self, ent_name):
        """
        Preprocesses entity name.

        :return: Preprocessed entity name.
        """
        ent_name = ent_name.strip()
        ent_name = trim1(ent_name)
        ent_name = ent_name.replace("&amp;", "&")
        ent_name = ent_name.replace("&quot;", '"')
        ent_name = ent_name.replace("_", " ")
        ent_name = first_letter_to_uppercase(ent_name)

        ent_name = self.wiki_redirect_ent_title(ent_name)
        return ent_name

    def ent_wiki_id_from_name(self, entity):
        """
        Preprocesses entity name and verifies if it exists in our KB.

        :return: Returns wikiID.
        """

        entity = self.preprocess_ent_name(entity)
        if not entity or (entity not in self.wiki_id_name_map["ent_name_to_id"]):
            return -1
        else:
            return self.wiki_id_name_map["ent_name_to_id"][entity]

    def wiki_redirect_ent_title(self, ent_name):
        """
        Verifies if entity name should redirect.

        :return: Returns wikipedia name
        """

        if ent_name in self.wiki_redirects_index:
            return self.wiki_redirects_index[ent_name]
        else:
            return ent_name

    def wiki_redirect_id(self, id):
        """
        Verifies if entity Id should redirect.

        :return: wikipedia Id
        """

        if id in self.wiki_redirects_id_index:
            return self.wiki_redirects_id_index[id]
        else:
            return id

    def generate_wiki_disambiguation_map(self):
        """
        Generates disambiguation index.

        :return: disambiguation index
        """

        wiki_disambiguation_index = {}
        path = os.path.join(self.base_url, "basic_data/wiki_disambiguation_pages.txt")

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                line = line.rstrip()
                parts = line.split("\t")
                assert int(parts[0])
                wiki_disambiguation_index[int(parts[0])] = 1
        return wiki_disambiguation_index

    def gen_wiki_name_map(self):
        """
        Generates wiki id/name and name/id index.

        :return: disambiguation index
        """

        wiki_id_name_map = {"ent_name_to_id": {}, "ent_id_to_name": {}}
        path = os.path.join(self.base_url, "basic_data/wiki_name_id_map.txt")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                parts = line.split("\t")

                ent_wiki_id = int(parts[1])
                ent_name = unquote(parts[0])

                if ent_wiki_id not in self.wiki_disambiguation_index:
                    wiki_id_name_map["ent_name_to_id"][ent_name] = ent_wiki_id
                    wiki_id_name_map["ent_id_to_name"][ent_wiki_id] = ent_name
        return wiki_id_name_map

    def generate_wiki_redirect_map(self):
        """
        Generates redirect index.

        :return: redirect index
        """
        wiki_redirects_index = {}
        wiki_redirects_id_index = {}
        path = os.path.join(self.base_url, "basic_data/wiki_redirects.txt")

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                line = line.rstrip()
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                parts[1] = unquote(parts[1])
                wiki_redirects_index[unquote(parts[0])] = parts[1]
                if len(parts) == 3:
                    wiki_redirects_id_index[int(parts[2])] = parts[1]
        return wiki_redirects_index, wiki_redirects_id_index
