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
        Links the query to the entity.

        Added optional parameter sentences_doc for completeness sake. If a user wishes to create his/her
        own MD system, it may deduce some form of a global context.
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
        print(sentence)
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
        """
        for exist_ngram, exist_pos in self.__ngrams_overlap:
            if ngram in exist_ngram:
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

    def __build_ngram(self, ngram, terms, start, i):
        quit = False

        for j in range(1, np.min([i, self.__n])):
            # Builds the sub-string.
            # If it is seperated by a trailing comma or whatever, then
            # it is most likely an end of the sentence.
            lookup = terms[start + j]
            if not re.match(r"^[_\W]+$", lookup):
                ngram += " {}".format(lookup)
            else:
                quit = True
                break
        return ngram, quit

    def __gen_ngrams(self, query):
        """Finds all n-grams of the query.
        :return: list of n-grams
        """
        terms = query.split()  # get_terms(query)
        ngrams = []

        for i in range(1, len(terms) + 1):  # number of words
            offset = 0
            for start in range(0, len(terms) - i + 1):  # start point
                ngram = terms[start]

                if re.match(r"^[_\W]+$", terms[start]):
                    # Invalid input
                    continue

                ngram, quit = self.__build_ngram(ngram, terms, start, i)

                if quit:
                    continue

                pos = self.__find_start_pos(query, start)
                end = self.__find_end_pos(ngram, query, pos)
                ngrams.append([ngram, pos, end])
        return ngrams
