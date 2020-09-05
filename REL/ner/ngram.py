import re
from collections import defaultdict, namedtuple

import numpy as np

from REL.db.generic import GenericLookup
from REL.mention_detection_base import MentionDetectionBase
from REL.ner import NERBase, Span
from REL.utils import preprocess_mention


class Cmns(NERBase, MentionDetectionBase):
    def __init__(self, base_url, wiki_version, n=5):
        self.__n = n
        super().__init__(base_url, wiki_version)

    def predict(self, sentence, sentences_doc):
        """

        Predicts using the ngram tagger.

        Note that we added sentences_doc for completeness sake as it may be used to infer some form of
        global context for custom MD modules.
        """
        self.__ngrams_overlap = []
        self.mentions = []

        self.rank_ens(sentence)

        # returns list of Span objects.
        return self.mentions

    def rank_ens(self, sentence):
        """
        Detects mention and rank entities for each mention
        """

        self.__get_ngrams(sentence)
        self.__recursive_rank_ens(self.__n)

    def __get_ngrams(self, sentence):
        """
        Returns n-grams grouped by length.
        """
        self.__ngrams = defaultdict(list)
        for ngram in self.__gen_ngrams(sentence):
            self.__ngrams[len(ngram[0].split())].append(ngram)

    def __recursive_rank_ens(self, n):
        """
        Generates list of entities for each mention in the query.
        The algorithm starts from the longest possible n-gram and gets all matched entities.
        If no entities found, the algorithm recurses and attempts find entities with (n-1)-gram.

        """
        if n == 0:
            return

        for ngram, pos, end in self.__ngrams[n]:
            if not self.__is_overlapping(ngram, pos):
                mention = self.preprocess_mention(ngram)
                freq = self.wiki_db.wiki(mention, "wiki", "freq")
                if freq:
                    self.mentions.append(Span(ngram, pos, end, freq, "#NGRAM#"))
                    self.__ngrams_overlap.append([ngram, pos])
        self.__recursive_rank_ens(n - 1)

    def __is_overlapping(self, ngram, pos_prop):
        """
        Checks whether the ngram is contained in one of the currently identified mentions.

        NOTE: Currently, this function is very basic and dependent on order of operations.

        """
        
        for exist_ngram, exist_pos in self.__ngrams_overlap:
            range_exist = set(range(exist_pos, exist_pos + len(exist_ngram)))
            range_new = set(range(pos_prop, pos_prop + len(ngram)))
            if len(range_exist.intersection(range_new)) > 0:
                return True
        return False

    def __find_end_pos(self, ngram, sent, start_pos):
        """
        Due to ngram detection, extra characters may be removed
        to improve performance. However, we still want to be able
        to find the original start and end position in the sentence.
        """
        splt = ngram.split()
        end = start_pos
        for s in splt:
            end = sent.find(s, end)
        end += len(s)

        return end

    def __find_start_pos(self, query, start):
        word_cnt = 0
        space_found = True
        pos = 0

        for char in query:
            if char.isspace():  # (chariss == ' ') and (char != '\t'):
                space_found = True
            elif space_found:
                space_found = False
                word_cnt += 1

            if word_cnt == (start + 1):
                break

            pos += 1
        return pos

    def __build_ngram(self, ngram, terms, start, end):
        quit = False

        for j in range(1, end+1):
            # Builds the sub-string.
            # If it is seperated by a trailing comma or whatever, then
            # we assume it is the end of the ngram.
            lookup = terms[start + j]

            if re.match(r"^[_\W]+$", lookup):
                quit = True
                break
            else:
                ngram += " {}".format(lookup)
        return ngram, quit

    def __gen_ngrams(self, query):
        """Finds all n-grams of the query.
        :return: list of n-grams
        """
        terms = query.split()
        ngrams = []

        for start_idx in range(0, len(terms)):  # number of words
            for end_idx in range(0, np.min([len(terms) - start_idx, self.__n])):  # start point
                ngram = terms[start_idx]
                
                if re.match(r"^[_\W]+$", ngram):
                    # Invalid input
                    continue
                ngram, quit = self.__build_ngram(ngram, terms, start_idx, end_idx)

                if quit:
                    continue

                pos = self.__find_start_pos(query, start_idx)
                end = self.__find_end_pos(ngram, query, pos)
                ngrams.append([ngram, pos, end])
        return ngrams
