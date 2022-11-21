import json
import os
import pickle as pkl
import re
import tarfile
import time
from pathlib import Path
from random import shuffle
from typing import Any, Dict
from urllib.parse import urlparse

import numpy as np
import pkg_resources
import torch
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch.autograd import Variable

import REL.utils as utils
from REL.db.generic import GenericLookup
from REL.mulrel_ranker import MulRelRanker, PreRank
from REL.training_datasets import TrainingEvaluationDatasets
from REL.vocabulary import Vocabulary


wiki_prefix = "en.wikipedia.org/wiki/"


class EntityDisambiguation:
    """
    Parent Entity Disambiguation class that directs the various subclasses used
    for the ED step.
    """
    def __init__(self, base_url, wiki_version, user_config, reset_embeddings=False):
        self.base_url = base_url
        self.wiki_version = wiki_version
        self.embeddings = {}
        self.config = self.__get_config(user_config)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.prerank_model = None
        self.model = None
        self.reset_embeddings = reset_embeddings
        self.emb = GenericLookup(
            "entity_word_embedding", os.path.join(base_url, wiki_version, "generated")
        )

        self.g_emb = GenericLookup("common_drawl", os.path.join(base_url, "generic"))
        test = self.g_emb.emb(["in"], "embeddings")[0]
        assert (
            test is not None
        ), "Glove embeddings in wrong folder..? Test embedding not found.."

        self.__load_embeddings()
        self.coref = TrainingEvaluationDatasets(base_url, wiki_version)
        self.prerank_model = PreRank(self.config).to(self.device)

        self.__max_conf = None

        # Load LR model for confidence.
        if os.path.exists(Path(self.config["model_path"]).parent / "lr_model.pkl"):
            with open(
                Path(self.config["model_path"]).parent / "lr_model.pkl",
                "rb",
            ) as f:
                self.model_lr = pkl.load(f)
        else:
            print("No LR model found, confidence scores ED will be set to zero.")
            self.model_lr = None

        if self.config["mode"] == "eval":
            print("Loading model from given path: {}".format(self.config["model_path"]))
            self.model = self.__load(self.config["model_path"])
        else:
            if reset_embeddings:
                raise Exception("You cannot train a model and reset the embeddings.")
            self.model = MulRelRanker(self.config, self.device).to(self.device)

    def __get_config(self, user_config):
        """
        User configuration that may overwrite default settings.

        :return: configuration used for ED.
        """

        default_config: Dict[str, Any] = {
            "mode": "train",
            "model_path": "./",
            "prerank_ctx_window": 50,
            "keep_p_e_m": 4,
            "keep_ctx_ent": 3,
            "ctx_window": 100,
            "tok_top_n": 25,
            "mulrel_type": "ment-norm",
            "n_rels": 3,
            "hid_dims": 100,
            "emb_dims": 300,
            "snd_local_ctx_window": 6,
            "dropout_rate": 0.3,
            "n_epochs": 1000,
            "dev_f1_change_lr": 0.915,
            "n_not_inc": 10,
            "eval_after_n_epochs": 5,
            "learning_rate": 1e-4,
            "margin": 0.01,
            "df": 0.5,
            "n_loops": 10,
            # 'freeze_embs': True,
            "n_cands_before_rank": 30,
            "first_head_uniforn": False,
            "use_pad_ent": True,
            "use_local": True,
            "use_local_only": False,
            "oracle": False,
        }

        default_config.update(user_config)
        config = default_config

        model_dict = json.loads(
            pkg_resources.resource_string("REL.models", "models.json")
        )
        model_path: str = config["model_path"]
        # load aliased url if it exists, else keep original string
        config["model_path"] = model_dict.get(model_path, model_path)

        if urlparse(str(config["model_path"])).scheme in ("http", "https"):
            model_path = utils.fetch_model(
                config["model_path"],
                cache_dir=Path("~/.rel_cache").expanduser(),
            )
            assert tarfile.is_tarfile(model_path), "Only tar-files are supported!"
            # make directory with name of tarfile (minus extension)
            # extract the files in the archive to that directory
            # assign config[model_path] accordingly
            with tarfile.open(model_path) as f:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(f, Path("~/.rel_cache").expanduser())
            # NOTE: use double stem to deal with e.g. *.tar.gz
            # this also handles *.tar correctly
            stem = Path(Path(model_path).stem).stem
            # NOTE: it is required that the model file(s) are named "model.state_dict"
            # and "model.config" if supplied, other names won't work.
            config["model_path"] = Path("~/.rel_cache").expanduser() / stem / "model"

        return config

    def __load_embeddings(self):
        """
        Initialised embedding dictionary and creates #UNK# token for respective embeddings.
        :return: -
        """
        self.__batch_embs = {}

        for name in ["snd", "entity", "word"]:
            # Init entity embeddings.
            self.embeddings["{}_seen".format(name)] = set()
            self.embeddings["{}_voca".format(name)] = Vocabulary()
            self.embeddings["{}_embeddings".format(name)] = None

            if name in ["word", "entity"]:
                # Add #UNK# token.
                self.embeddings["{}_voca".format(name)].add_to_vocab("#UNK#")
                e = self.emb.emb(["#{}/UNK#".format(name.upper())], "embeddings")[0]

                assert e is not None, "#UNK# token not found for {} in db".format(name)

                self.__batch_embs[name] = []
                self.__batch_embs[name].append(torch.tensor(e))
            else:
                # For Glove the #UNK# token was randomly initialised as can be seen. We added this to
                # our generated database for reproducability. Author also reports no significant difference
                # in using the mean of the vector or a randomly intialised vector for the glove embeddings.
                # https://github.com/lephong/mulrel-nel/issues/21
                self.embeddings["{}_voca".format(name)].add_to_vocab("#UNK#")
                e = self.g_emb.emb(["#SND/UNK#"], "embeddings")[0]

                assert e is not None, "#UNK# token not found for {} in db".format(name)

                self.__batch_embs[name] = []
                self.__batch_embs[name].append(torch.tensor(e))

    def train(self, org_train_dataset, org_dev_datasets):
        """
        Responsible for training the ED model.

        :return: -
        """

        train_dataset = self.get_data_items(org_train_dataset, "train", predict=False)
        dev_datasets = []
        for dname, data in org_dev_datasets.items():
            dev_datasets.append((dname, self.get_data_items(data, dname, predict=True)))

        print("Creating optimizer")
        optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config["learning_rate"],
        )
        best_f1 = -1
        not_better_count = 0
        eval_after_n_epochs = self.config["eval_after_n_epochs"]

        for e in range(self.config["n_epochs"]):
            shuffle(train_dataset)

            total_loss = 0
            for dc, batch in enumerate(train_dataset):  # each document is a minibatch
                self.model.train()
                optimizer.zero_grad()

                # convert data items to pytorch inputs
                token_ids = [
                    m["context"][0] + m["context"][1]
                    if len(m["context"][0]) + len(m["context"][1]) > 0
                    else [self.embeddings["word_voca"].unk_id]
                    for m in batch
                ]
                s_ltoken_ids = [m["snd_ctx"][0] for m in batch]
                s_rtoken_ids = [m["snd_ctx"][1] for m in batch]
                s_mtoken_ids = [m["snd_ment"] for m in batch]

                entity_ids = Variable(
                    torch.LongTensor([m["selected_cands"]["cands"] for m in batch]).to(
                        self.device
                    )
                )
                true_pos = Variable(
                    torch.LongTensor(
                        [m["selected_cands"]["true_pos"] for m in batch]
                    ).to(self.device)
                )
                p_e_m = Variable(
                    torch.FloatTensor([m["selected_cands"]["p_e_m"] for m in batch]).to(
                        self.device
                    )
                )
                entity_mask = Variable(
                    torch.FloatTensor([m["selected_cands"]["mask"] for m in batch]).to(
                        self.device
                    )
                )

                token_ids, token_mask = utils.make_equal_len(
                    token_ids, self.embeddings["word_voca"].unk_id
                )
                s_ltoken_ids, s_ltoken_mask = utils.make_equal_len(
                    s_ltoken_ids, self.embeddings["snd_voca"].unk_id, to_right=False
                )
                s_rtoken_ids, s_rtoken_mask = utils.make_equal_len(
                    s_rtoken_ids, self.embeddings["snd_voca"].unk_id
                )
                s_rtoken_ids = [l[::-1] for l in s_rtoken_ids]
                s_rtoken_mask = [l[::-1] for l in s_rtoken_mask]
                s_mtoken_ids, s_mtoken_mask = utils.make_equal_len(
                    s_mtoken_ids, self.embeddings["snd_voca"].unk_id
                )

                token_ids = Variable(torch.LongTensor(token_ids).to(self.device))
                token_mask = Variable(torch.FloatTensor(token_mask).to(self.device))

                # too ugly but too lazy to fix it
                self.model.s_ltoken_ids = Variable(
                    torch.LongTensor(s_ltoken_ids).to(self.device)
                )
                self.model.s_ltoken_mask = Variable(
                    torch.FloatTensor(s_ltoken_mask).to(self.device)
                )
                self.model.s_rtoken_ids = Variable(
                    torch.LongTensor(s_rtoken_ids).to(self.device)
                )
                self.model.s_rtoken_mask = Variable(
                    torch.FloatTensor(s_rtoken_mask).to(self.device)
                )
                self.model.s_mtoken_ids = Variable(
                    torch.LongTensor(s_mtoken_ids).to(self.device)
                )
                self.model.s_mtoken_mask = Variable(
                    torch.FloatTensor(s_mtoken_mask).to(self.device)
                )

                scores, ent_scores = self.model.forward(
                    token_ids,
                    token_mask,
                    entity_ids,
                    entity_mask,
                    p_e_m,
                    self.embeddings,
                    gold=true_pos.view(-1, 1),
                )
                loss = self.model.loss(scores, true_pos)
                # loss = self.model.prob_loss(scores, true_pos)
                loss.backward()
                optimizer.step()
                self.model.regularize(max_norm=100)

                loss = loss.cpu().data.numpy()
                total_loss += loss
                print(
                    "epoch",
                    e,
                    "%0.2f%%" % (dc / len(train_dataset) * 100),
                    loss,
                    end="\r",
                )

            print("epoch", e, "total loss", total_loss, total_loss / len(train_dataset))

            if (e + 1) % eval_after_n_epochs == 0:
                dev_f1 = 0
                for dname, data in dev_datasets:
                    predictions = self.__predict(data)
                    f1, recall, precision, _ = self.__eval(
                        org_dev_datasets[dname], predictions
                    )
                    print(
                        dname,
                        utils.tokgreen(
                            "Micro F1: {}, Recall: {}, Precision: {}".format(
                                f1, recall, precision
                            )
                        ),
                    )

                    if dname == "aida_testA":
                        dev_f1 = f1

                if (
                    self.config["learning_rate"] == 1e-4
                    and dev_f1 >= self.config["dev_f1_change_lr"]
                ):
                    eval_after_n_epochs = 2
                    best_f1 = dev_f1
                    not_better_count = 0

                    self.config["learning_rate"] = 1e-5
                    print("change learning rate to", self.config["learning_rate"])
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = self.config["learning_rate"]

                if dev_f1 < best_f1:
                    not_better_count += 1
                    print("Not improving", not_better_count)
                else:
                    not_better_count = 0
                    best_f1 = dev_f1
                    print("save model to", self.config["model_path"])
                    self.__save(self.config["model_path"])

                if not_better_count == self.config["n_not_inc"]:
                    break

    def evaluate(self, datasets):
        """
        Parent function r esponsible for evaluating the ED model during the ED step. Note that
        this is different from predict as this requires ground truth entities to be present.

        :return: -
        """

        dev_datasets = []
        for dname, data in list(datasets.items()):
            start = time.time()
            dev_datasets.append((dname, self.get_data_items(data, dname, predict=True)))

        for dname, data in dev_datasets:
            predictions = self.__predict(data)
            f1, recall, precision, total_nil = self.__eval(datasets[dname], predictions)
            print(
                dname,
                utils.tokgreen(
                    "Micro F1: {}, Recall: {}, Precision: {}".format(
                        f1, recall, precision
                    )
                ),
            )
            print("Total NIL: {}".format(total_nil))
            print("----------------------------------")

    def __create_dataset_LR(self, datasets, predictions, dname):
        X = []
        y = []
        meta = []
        for doc, preds in predictions.items():
            gt_doc = [c["gold"][0] for c in datasets[dname][doc]]
            for pred, gt in zip(preds, gt_doc):
                scores = [float(x) for x in pred["scores"]]
                cands = pred["candidates"]

                # Build classes
                for i, c in enumerate(cands):
                    if c == "#UNK#":
                        continue

                    X.append([scores[i]])
                    meta.append([doc, gt, c])
                    if gt == c:
                        y.append(1.0)
                    else:
                        y.append(0.0)

        return np.array(X), np.array(y), np.array(meta)

    def train_LR(self, datasets, model_path_lr, store_offline=True, threshold=0.3):
        """
        Function that applies LR in an attempt to get confidence scores. Recall should be high,
        because if it is low than we would have ignored a corrrect entity.

        :return: -
        """
        print(os.path.join(model_path_lr, "lr_model.pkl"))

        train_dataset = self.get_data_items(
            datasets["aida_train"], "train", predict=False
        )

        dev_datasets = []
        for dname, data in list(datasets.items()):
            if dname == "aida_train":
                continue
            dev_datasets.append((dname, self.get_data_items(data, dname, predict=True)))

        model = LogisticRegression()

        predictions = self.__predict(train_dataset, eval_raw=True)
        X, y, meta = self.__create_dataset_LR(datasets, predictions, "aida_train")
        model.fit(X, y)

        for dname, data in dev_datasets:
            predictions = self.__predict(data, eval_raw=True)
            X, y, meta = self.__create_dataset_LR(datasets, predictions, dname)
            preds = model.predict_proba(X)
            preds = np.array([x[1] for x in preds])

            decisions = (preds >= threshold).astype(int)

            print(
                utils.tokgreen("{}, F1-score: {}".format(dname, f1_score(y, decisions)))
            )

        if store_offline:
            path = os.path.join(model_path_lr, "lr_model.pkl")
            with open(path, "wb") as handle:
                pkl.dump(model, handle, protocol=pkl.HIGHEST_PROTOCOL)

    def predict(self, data):
        """
        Parent function responsible for predicting on any raw text as input. This does not require ground
        truth entities to be present.

        :return: predictions and time taken for the ED step.
        """

        self.coref.with_coref(data)
        data = self.get_data_items(data, "raw", predict=True)
        predictions, timing = self.__predict(data, include_timing=True, eval_raw=True)

        return predictions, timing

    def __compute_confidence_legacy(self, scores, preds):
        """
        LEGACY

        :return:
        """
        confidence_scores = []

        for score, pred in zip(scores, preds):
            loss = 0
            for j in range(len(score)):
                if j == pred:
                    continue
                loss += max(
                    0, score[j].item() - score[pred].item() + self.config["margin"]
                )
            if not self.__max_conf:
                self.__max_conf = (
                    self.config["keep_ctx_ent"] + self.config["keep_p_e_m"] - 1
                ) * self.config["margin"]
            conf = 1 - (loss / self.__max_conf)
            confidence_scores.append(conf)

        return confidence_scores

    def __compute_confidence(self, scores, preds):
        """
        Uses LR to find confidence scores for given ED outputs.

        :return:
        """
        X = np.array([[score[pred]] for score, pred in zip(scores, preds)])
        if self.model_lr:
            preds = self.model_lr.predict_proba(X)
            confidence_scores = [x[1] for x in preds]
        else:
            confidence_scores = [0.0 for _ in scores]
        return confidence_scores

    def __predict(self, data, include_timing=False, eval_raw=False):
        """
        Uses the trained model to make predictions of individual batches (i.e. documents).

        :return: predictions and time taken for the ED step.
        """

        predictions = {items[0]["doc_name"]: [] for items in data}
        self.model.eval()

        timing = []

        for batch in data:  # each document is a minibatch

            start = time.time()

            token_ids = [
                m["context"][0] + m["context"][1]
                if len(m["context"][0]) + len(m["context"][1]) > 0
                else [self.embeddings["word_voca"].unk_id]
                for m in batch
            ]
            s_ltoken_ids = [m["snd_ctx"][0] for m in batch]
            s_rtoken_ids = [m["snd_ctx"][1] for m in batch]
            s_mtoken_ids = [m["snd_ment"] for m in batch]

            entity_ids = Variable(
                torch.LongTensor([m["selected_cands"]["cands"] for m in batch]).to(
                    self.device
                )
            )
            p_e_m = Variable(
                torch.FloatTensor([m["selected_cands"]["p_e_m"] for m in batch]).to(
                    self.device
                )
            )
            entity_mask = Variable(
                torch.FloatTensor([m["selected_cands"]["mask"] for m in batch]).to(
                    self.device
                )
            )
            true_pos = Variable(
                torch.LongTensor([m["selected_cands"]["true_pos"] for m in batch]).to(
                    self.device
                )
            )

            token_ids, token_mask = utils.make_equal_len(
                token_ids, self.embeddings["word_voca"].unk_id
            )
            s_ltoken_ids, s_ltoken_mask = utils.make_equal_len(
                s_ltoken_ids, self.embeddings["snd_voca"].unk_id, to_right=False
            )
            s_rtoken_ids, s_rtoken_mask = utils.make_equal_len(
                s_rtoken_ids, self.embeddings["snd_voca"].unk_id
            )
            s_rtoken_ids = [l[::-1] for l in s_rtoken_ids]
            s_rtoken_mask = [l[::-1] for l in s_rtoken_mask]
            s_mtoken_ids, s_mtoken_mask = utils.make_equal_len(
                s_mtoken_ids, self.embeddings["snd_voca"].unk_id
            )

            token_ids = Variable(torch.LongTensor(token_ids).to(self.device))
            token_mask = Variable(torch.FloatTensor(token_mask).to(self.device))

            self.model.s_ltoken_ids = Variable(
                torch.LongTensor(s_ltoken_ids).to(self.device)
            )
            self.model.s_ltoken_mask = Variable(
                torch.FloatTensor(s_ltoken_mask).to(self.device)
            )
            self.model.s_rtoken_ids = Variable(
                torch.LongTensor(s_rtoken_ids).to(self.device)
            )
            self.model.s_rtoken_mask = Variable(
                torch.FloatTensor(s_rtoken_mask).to(self.device)
            )
            self.model.s_mtoken_ids = Variable(
                torch.LongTensor(s_mtoken_ids).to(self.device)
            )
            self.model.s_mtoken_mask = Variable(
                torch.FloatTensor(s_mtoken_mask).to(self.device)
            )

            scores, ent_scores = self.model.forward(
                token_ids,
                token_mask,
                entity_ids,
                entity_mask,
                p_e_m,
                self.embeddings,
                gold=true_pos.view(-1, 1),
            )
            pred_ids = torch.argmax(scores, axis=1)
            scores = scores.cpu().data.numpy()

            confidence_scores = self.__compute_confidence(scores, pred_ids)
            pred_ids = np.argmax(scores, axis=1)

            if not eval_raw:
                pred_entities = [
                    m["selected_cands"]["named_cands"][i]
                    if m["selected_cands"]["mask"][i] == 1
                    else (
                        m["selected_cands"]["named_cands"][0]
                        if m["selected_cands"]["mask"][0] == 1
                        else "NIL"
                    )
                    for (i, m) in zip(pred_ids, batch)
                ]
                doc_names = [m["doc_name"] for m in batch]

                for dname, entity in zip(doc_names, pred_entities):
                    predictions[dname].append({"pred": (entity, 0.0)})

            else:
                pred_entities = [
                    [
                        m["selected_cands"]["named_cands"][i],
                        m["raw"]["mention"],
                        m["selected_cands"]["named_cands"],
                        s,
                        cs,
                        m["selected_cands"]["mask"],
                    ]
                    if m["selected_cands"]["mask"][i] == 1
                    else (
                        [
                            m["selected_cands"]["named_cands"][0],
                            m["raw"]["mention"],
                            m["selected_cands"]["named_cands"],
                            s,
                            cs,
                            m["selected_cands"]["mask"],
                        ]
                        if m["selected_cands"]["mask"][0] == 1
                        else [
                            "NIL",
                            m["raw"]["mention"],
                            m["selected_cands"]["named_cands"],
                            s,
                            cs,
                            m["selected_cands"]["mask"],
                        ]
                    )
                    for (i, m, s, cs) in zip(pred_ids, batch, scores, confidence_scores)
                ]
                doc_names = [m["doc_name"] for m in batch]

                for dname, entity in zip(doc_names, pred_entities):
                    if entity[0] != "NIL":
                        predictions[dname].append(
                            {
                                "mention": entity[1],
                                "prediction": entity[0],
                                "candidates": entity[2],
                                "conf_ed": entity[4],
                                "scores": list([str(x) for x in entity[3]]),
                            }
                        )

                    else:
                        predictions[dname].append(
                            {
                                "mention": entity[1],
                                "prediction": entity[0],
                                "candidates": entity[2],
                                "scores": [],
                            }
                        )

            timing.append(time.time() - start)
        if include_timing:
            return predictions, timing
        else:
            return predictions

    def prerank(self, dataset, dname, predict=False):
        """
        Responsible for preranking the set of possible candidates using both context and p(e|m) scores.
        :return: dataset with, by default, max 3 + 4 candidates per mention.
        """
        new_dataset = []
        has_gold = 0
        total = 0

        for content in dataset:
            items = []
            if self.config["keep_ctx_ent"] > 0:
                # rank the candidates by ntee scores
                lctx_ids = [
                    m["context"][0][
                        max(
                            len(m["context"][0])
                            - self.config["prerank_ctx_window"] // 2,
                            0,
                        ) :
                    ]
                    for m in content
                ]
                rctx_ids = [
                    m["context"][1][
                        : min(
                            len(m["context"][1]), self.config["prerank_ctx_window"] // 2
                        )
                    ]
                    for m in content
                ]
                ment_ids = [[] for m in content]
                token_ids = [
                    l + m + r
                    if len(l) + len(r) > 0
                    else [self.embeddings["word_voca"].unk_id]
                    for l, m, r in zip(lctx_ids, ment_ids, rctx_ids)
                ]

                entity_ids = [m["cands"] for m in content]
                entity_ids = Variable(torch.LongTensor(entity_ids).to(self.device))

                entity_mask = [m["mask"] for m in content]
                entity_mask = Variable(torch.FloatTensor(entity_mask).to(self.device))

                token_ids, token_offsets = utils.flatten_list_of_lists(token_ids)
                token_offsets = Variable(
                    torch.LongTensor(token_offsets).to(self.device)
                )
                token_ids = Variable(torch.LongTensor(token_ids).to(self.device))

                entity_names = [m["named_cands"] for m in content]  # named_cands

                log_probs = self.prerank_model.forward(
                    token_ids, token_offsets, entity_ids, self.embeddings, self.emb
                )

                # Entity mask makes sure that the UNK entities are zero.
                log_probs = (log_probs * entity_mask).add_((entity_mask - 1).mul_(1e10))
                _, top_pos = torch.topk(log_probs, dim=1, k=self.config["keep_ctx_ent"])
                top_pos = top_pos.data.cpu().numpy()

            else:
                top_pos = [[]] * len(content)

            # select candidats: mix between keep_ctx_ent best candidates (ntee scores) with
            # keep_p_e_m best candidates (p_e_m scores)
            for i, m in enumerate(content):
                sm = {
                    "cands": [],
                    "named_cands": [],
                    "p_e_m": [],
                    "mask": [],
                    "true_pos": -1,
                }
                m["selected_cands"] = sm

                selected = set(top_pos[i])
                idx = 0
                while (
                    len(selected)
                    < self.config["keep_ctx_ent"] + self.config["keep_p_e_m"]
                ):
                    if idx not in selected:
                        selected.add(idx)
                    idx += 1

                selected = sorted(list(selected))
                for idx in selected:
                    sm["cands"].append(m["cands"][idx])
                    sm["named_cands"].append(m["named_cands"][idx])
                    sm["p_e_m"].append(m["p_e_m"][idx])
                    sm["mask"].append(m["mask"][idx])
                    if idx == m["true_pos"]:
                        sm["true_pos"] = len(sm["cands"]) - 1

                if not predict:
                    if sm["true_pos"] == -1:
                        continue

                items.append(m)
                if sm["true_pos"] >= 0:
                    has_gold += 1
                total += 1

                if predict:
                    # only for oracle model, not used for eval
                    if sm["true_pos"] == -1:
                        sm[
                            "true_pos"
                        ] = 0  # a fake gold, happens only 2%, but avoid the non-gold

            if len(items) > 0:
                new_dataset.append(items)

        # if total > 0
        if dname != "raw":
            print("Recall for {}: {}".format(dname, has_gold / total))
            print("-----------------------------------------------")
        return new_dataset

    def __update_embeddings(self, emb_name, embs):
        """
        Responsible for updating the dictionaries with their respective word, entity and snd (GloVe) embeddings.

        :return: -
        """

        embs = embs.to(self.device)

        if self.embeddings["{}_embeddings".format(emb_name)]:
            new_weights = torch.cat(
                (self.embeddings["{}_embeddings".format(emb_name)].weight, embs)
            )
        else:
            new_weights = embs

        # Weights are now updated, so we create a new Embedding layer.
        layer = torch.nn.Embedding(
            self.embeddings["{}_voca".format(emb_name)].size(), self.config["emb_dims"]
        )
        layer.weight = torch.nn.Parameter(new_weights)
        layer.grad = False
        self.embeddings["{}_embeddings".format(emb_name)] = layer
        if emb_name == "word":
            layer = torch.nn.EmbeddingBag(
                self.embeddings["{}_voca".format(emb_name)].size(),
                self.config["emb_dims"],
            )
            layer.weight = torch.nn.Parameter(new_weights)

            layer.requires_grad = False
            self.embeddings["{}_embeddings_bag".format(emb_name)] = layer

        del new_weights

    def __embed_words(self, words_filt, name, table_name="embeddings"):
        """
        Responsible for retrieving embeddings using the given sqlite3 database.

        :return: -
        """

        # Returns None if not in db.
        if table_name == "glove":
            embs = self.g_emb.emb(words_filt, "embeddings")
        else:
            embs = self.emb.emb(words_filt, table_name)

        # Now we go over the embs and see which one is None. Order is preserved.
        for e, c in zip(embs, words_filt):
            if name == "entity":
                c = c.replace("ENTITY/", "")
            self.embeddings["{}_seen".format(name)].add(c)
            if e is not None:
                # Embedding exists, so we add it.
                self.embeddings["{}_voca".format(name)].add_to_vocab(c)
                self.__batch_embs[name].append(torch.tensor(e))

    def get_data_items(self, dataset, dname, predict=False):
        """
        Responsible for formatting dataset. Triggers the preranking function.

        :return: preranking function.
        """
        data = []

        if self.reset_embeddings:
            # If user wants to reset, he can do this here, right before loading a new dataset.
            self.__load_embeddings()

        for doc_name, content in dataset.items():
            items = []
            if len(content) == 0:
                continue
            conll_doc = content[0].get("conll_doc", None)
            for m in content:
                named_cands = [c[0] for c in m["candidates"]]
                p_e_m = [min(1.0, max(1e-3, c[1])) for c in m["candidates"]]

                try:
                    true_pos = named_cands.index(m["gold"][0])
                    p = p_e_m[true_pos]
                except:
                    true_pos = -1

                # Get all words and check for embeddings.
                named_cands = named_cands[
                    : min(self.config["n_cands_before_rank"], len(named_cands))
                ]

                # Candidate list per mention.
                named_cands_filt = set(
                    [
                        "ENTITY/" + item
                        for item in named_cands
                        if item not in self.embeddings["entity_seen"]
                    ]
                )

                self.__embed_words(named_cands_filt, "entity", "embeddings")

                # Use re.split() to make sure that special characters are considered.
                lctx = [
                    x for x in re.split("(\W)", m["context"][0].strip()) if x != " "
                ]  # .split()
                rctx = [
                    x for x in re.split("(\W)", m["context"][1].strip()) if x != " "
                ]  # split()

                words_filt = set(
                    [
                        item
                        for item in lctx + rctx
                        if item not in self.embeddings["word_seen"]
                    ]
                )

                self.__embed_words(words_filt, "word", "embeddings")

                snd_lctx = m["sentence"][: m["pos"]].strip().split()
                snd_lctx = [
                    t for t in snd_lctx[-self.config["snd_local_ctx_window"] // 2 :]
                ]

                snd_rctx = m["sentence"][m["end_pos"] :].strip().split()
                snd_rctx = [
                    t for t in snd_rctx[: self.config["snd_local_ctx_window"] // 2]
                ]

                snd_ment = m["ngram"].strip().split()

                words_filt = set(
                    [
                        item
                        for item in snd_lctx + snd_rctx + snd_ment
                        if item not in self.embeddings["snd_seen"]
                    ]
                )

                self.__embed_words(words_filt, "snd", "glove")

                p_e_m = p_e_m[: min(self.config["n_cands_before_rank"], len(p_e_m))]

                if true_pos >= len(named_cands):
                    if not predict:
                        true_pos = len(named_cands) - 1
                        p_e_m[-1] = p
                        named_cands[-1] = m["gold"][0]
                    else:
                        true_pos = -1
                cands = [
                    self.embeddings["entity_voca"].get_id(
                        # ("" if self.generic else wiki_prefix) + c
                        c
                    )
                    for c in named_cands
                ]

                mask = [1.0] * len(cands)
                if len(cands) == 0 and not predict:
                    continue
                elif len(cands) < self.config["n_cands_before_rank"]:
                    cands += [self.embeddings["entity_voca"].unk_id] * (
                        self.config["n_cands_before_rank"] - len(cands)
                    )
                    named_cands += [Vocabulary.unk_token] * (
                        self.config["n_cands_before_rank"] - len(named_cands)
                    )
                    p_e_m += [1e-8] * (self.config["n_cands_before_rank"] - len(p_e_m))
                    mask += [0.0] * (self.config["n_cands_before_rank"] - len(mask))

                lctx_ids = [
                    self.embeddings["word_voca"].get_id(t)
                    for t in lctx
                    if utils.is_important_word(t)
                ]

                lctx_ids = [
                    tid
                    for tid in lctx_ids
                    if tid != self.embeddings["word_voca"].unk_id
                ]
                lctx_ids = lctx_ids[
                    max(0, len(lctx_ids) - self.config["ctx_window"] // 2) :
                ]

                rctx_ids = [
                    self.embeddings["word_voca"].get_id(t)
                    for t in rctx
                    if utils.is_important_word(t)
                ]
                rctx_ids = [
                    tid
                    for tid in rctx_ids
                    if tid != self.embeddings["word_voca"].unk_id
                ]
                rctx_ids = rctx_ids[
                    : min(len(rctx_ids), self.config["ctx_window"] // 2)
                ]

                ment = m["mention"].strip().split()
                ment_ids = [
                    self.embeddings["word_voca"].get_id(t)
                    for t in ment
                    if utils.is_important_word(t)
                ]
                ment_ids = [
                    tid
                    for tid in ment_ids
                    if tid != self.embeddings["word_voca"].unk_id
                ]

                m["sent"] = " ".join(lctx + rctx)

                # Secondary local context.
                snd_lctx = [self.embeddings["snd_voca"].get_id(t) for t in snd_lctx]
                snd_rctx = [self.embeddings["snd_voca"].get_id(t) for t in snd_rctx]
                snd_ment = [self.embeddings["snd_voca"].get_id(t) for t in snd_ment]

                # This is only used for the original embeddings, now they are never empty.
                if len(snd_lctx) == 0:
                    snd_lctx = [self.embeddings["snd_voca"].unk_id]
                if len(snd_rctx) == 0:
                    snd_rctx = [self.embeddings["snd_voca"].unk_id]
                if len(snd_ment) == 0:
                    snd_ment = [self.embeddings["snd_voca"].unk_id]

                items.append(
                    {
                        "context": (lctx_ids, rctx_ids),
                        "snd_ctx": (snd_lctx, snd_rctx),
                        "ment_ids": ment_ids,
                        "snd_ment": snd_ment,
                        "cands": cands,
                        "named_cands": named_cands,
                        "p_e_m": p_e_m,
                        "mask": mask,
                        "true_pos": true_pos,
                        "doc_name": doc_name,
                        "raw": m,
                    }
                )

            if len(items) > 0:
                # note: this shouldn't affect the order of prediction because we use doc_name to add predicted entities,
                # and we don't shuffle the data for prediction
                if len(items) > 100:
                    # print(len(items))
                    for k in range(0, len(items), 100):
                        data.append(items[k : min(len(items), k + 100)])
                else:
                    data.append(items)

        # Update batch
        for n in ["word", "entity", "snd"]:
            if self.__batch_embs[n]:
                self.__batch_embs[n] = torch.stack(self.__batch_embs[n])
                self.__update_embeddings(n, self.__batch_embs[n])
                self.__batch_embs[n] = []

        return self.prerank(data, dname, predict)

    def __eval(self, testset, system_pred):
        """
        Responsible for evaluating data points, which is solely used for the local ED step.

        :return: F1, Recall, Precision and number of mentions for which we have no valid candidate.
        """
        gold = []
        pred = []

        for doc_name, content in testset.items():
            if len(content) == 0:
                continue
            gold += [c["gold"][0] for c in content]
            pred += [c["pred"][0] for c in system_pred[doc_name]]

        true_pos = 0
        total_nil = 0
        for g, p in zip(gold, pred):
            if p == "NIL":
                total_nil += 1
            if g == p and p != "NIL":
                true_pos += 1

        precision = true_pos / len([p for p in pred if p != "NIL"])
        recall = true_pos / len(gold)
        f1 = 2 * precision * recall / (precision + recall)
        return f1, recall, precision, total_nil

    def __save(self, path):
        """
        Responsible for storing the trained model during optimisation.

        :return: -.
        """
        torch.save(self.model.state_dict(), "{}.state_dict".format(path))
        with open("{}.config".format(path), "w") as f:
            json.dump(self.config, f)

    def __load(self, path):
        """
        Responsible for loading a trained model and its respective config. Note that this config cannot be
        overwritten. If required, this behavior may be modified in future releases.

        :return: model
        """

        if os.path.exists("{}.config".format(path)):
            with open("{}.config".format(path), "r") as f:
                temp = self.config["model_path"]
                self.config = json.load(f)
                self.config["model_path"] = temp
        else:
            print(
                "No configuration file found at {}, default settings will be used.".format(
                    "{}.config".format(path)
                )
            )

        model = MulRelRanker(self.config, self.device).to(
            self.device
        )  # , self.embeddings

        if not torch.cuda.is_available():
            model.load_state_dict(
                torch.load(
                    "{}{}".format(self.config["model_path"], ".state_dict"),
                    map_location=torch.device("cpu"),
                )
            )
        else:
            model.load_state_dict(
                torch.load("{}{}".format(self.config["model_path"], ".state_dict"))
            )
        return model
