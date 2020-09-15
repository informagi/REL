import os
import pickle

"""
Class responsible for loading training/evaluation datasets for local ED.
"""


class TrainingEvaluationDatasets:
    """
    reading dataset from CoNLL dataset, extracted by https://github.com/dalab/deep-ed/
    """

    def __init__(self, base_url, wiki_version):
        self.person_names = self.__load_person_names(
            os.path.join(base_url, "generic/p_e_m_data/persons.txt")
        )
        self.base_url = os.path.join(base_url, wiki_version)

    def load(self):
        """
        Loads respective datasets and processes coreferences.

        :return: Returns training/evaluation datasets.
        """
        datasets = {}
        for ds in [
            "aida_train",
            "aida_testA",
            "aida_testB",
            "wned-ace2004",
            "wned-aquaint",
            "wned-clueweb",
            "wned-msnbc",
            "wned-wikipedia",
        ]:

            print("Loading {}".format(ds))
            datasets[ds] = self.__read_pickle_file(
                os.path.join(self.base_url, "generated/test_train_data/", f"{ds}.pkl")
            )

            if ds == "wned-wikipedia":
                if "Jiří_Třanovský" in datasets[ds]:
                    del datasets[ds]["Jiří_Třanovský"]
                if "Jiří_Třanovský Jiří_Třanovský" in datasets[ds]:
                    del datasets[ds]["Jiří_Třanovský Jiří_Třanovský"]

            self.with_coref(datasets[ds])

        return datasets

    def __read_pickle_file(self, path):
        """
        Reads pickle file.

        :return: Dataset
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        return data

    def __load_person_names(self, path):
        """
        Loads person names to find coreferences.

        :return: set of names.
        """

        data = []
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                data.append(line.strip().replace(" ", "_"))
        return set(data)

    def __find_coref(self, ment, mentlist):
        """
        Attempts to find coreferences

        :return: coreferences
        """

        cur_m = ment["mention"].lower()
        coref = []
        for m in mentlist:
            if (
                len(m["candidates"]) == 0
                or m["candidates"][0][0] not in self.person_names
            ):
                continue

            mention = m["mention"].lower()
            start_pos = mention.find(cur_m)
            if start_pos == -1 or mention == cur_m:
                continue

            end_pos = start_pos + len(cur_m) - 1
            if (start_pos == 0 or mention[start_pos - 1] == " ") and (
                end_pos == len(mention) - 1 or mention[end_pos + 1] == " "
            ):
                coref.append(m)

        return coref

    def with_coref(self, dataset):
        """
        Parent function that checks if there are coreferences in the given dataset.

        :return: dataset
        """

        for data_name, content in dataset.items():
            for cur_m in content:
                coref = self.__find_coref(cur_m, content)
                if coref is not None and len(coref) > 0:
                    cur_cands = {}
                    for m in coref:
                        for c, p in m["candidates"]:
                            cur_cands[c] = cur_cands.get(c, 0) + p
                    for c in cur_cands.keys():
                        cur_cands[c] /= len(coref)
                    cur_m["candidates"] = sorted(
                        list(cur_cands.items()), key=lambda x: x[1]
                    )[::-1]
