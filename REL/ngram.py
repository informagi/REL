import re
from collections import defaultdict

import numpy as np

from REL.db.generic import GenericLookup
from REL.utils import preprocess_mention

class Token:
    def __init__(self, text, start_pos, end_pos, score, tag):
        self.text = text
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.score = score
        self.tag = tag

class Cmns(object):
    def __init__(self, base_url, wiki_subfolder, n=5):
        self.__n = n
        self.wiki_db = GenericLookup(
            "entity_word_embedding",
            "{}/{}/generated/".format(base_url, wiki_subfolder),
        )
    def predict(self, sentence, sentences_doc):
        """Links the query to the entity.
        dictionary {mention: (en_id, score), ..}

        Have added optional parameter sentences_doc for completeness sake. If a user wishes to create his/her
        own MD system that also utilizes some form of a global context, then this may be a requirement.
        """
        self.__ngrams_overlap = []
        self.mentions = []

        self.rank_ens(sentence)

        # returns list of Token objects.
        return self.mentions

    def rank_ens(self, sentence):
        """Detects mention and rank entities for each mention"""

        self.__get_ngrams(sentence)
        self.__recursive_rank_ens(self.__n)

    def __get_ngrams(self, sentence):
        """Returns n-grams grouped by length.
        :return: dictionary {1:["xx", ...], 2: ["xx yy", ...], ...}
        """
        self.__ngrams = defaultdict(list)
        for ngram in self.__gen_ngrams(sentence):
            self.__ngrams[len(ngram[0].split())].append(ngram)

    def __recursive_rank_ens(self, n):
        """Generates list of entities for each mention in the query.
        The algorithm starts from the longest possible n-gram and gets all matched entities.
        If no entities found, the algorithm recurses and attempts find entities with (n-1)-gram.
        :param n: length of n-gram
        :return: dictionary {(dbp_uri, fb_id):commonness, ..}
        """
        if n == 0:
            return

        for ngram, pos, end in self.__ngrams[n]:
            if not self.__is_overlapping(ngram, pos):
                #TODO: Redundant call.
                mention = preprocess_mention(ngram, self.wiki_db)
                freq = self.wiki_db.wiki(mention, "wiki", "freq")
                if freq:
                    self.mentions.append(Token(ngram, pos, end, freq, '#NGRAM#'))
                    self.__ngrams_overlap.append([ngram, pos])
        self.__recursive_rank_ens(n - 1)

    def __is_overlapping(self, ngram, pos_prop):
        """Checks whether the ngram is contained in one of the currently identified mentions."""
        for exist_ngram, exist_pos in self.__ngrams_overlap:
            if ngram in exist_ngram:
                range_exist = set(range(exist_pos, exist_pos + len(exist_ngram)))
                range_new = set(range(pos_prop, pos_prop + len(ngram)))
                if len(range_exist.intersection(range_new)) > 0:
                    return True
        return False

    def __find_end_pos(self, ngram, sent, start_pos):
        '''
        Due to ngram detection, extra characters may be removed
        to improve performance. However, we still want to be able
        to find the original start and end position in the sentence.
        '''
        splt = ngram.split()
        end = start_pos
        for s in splt:
            end = sent.find(s, end)
        end += len(s)

        return end

    def __gen_ngrams(self, query):
        """Finds all n-grams of the query.
        :return: list of n-grams
        """
        #TODO: UGLY code ... can probably use word.find() + offset or something.
        terms = query.split()  # get_terms(query)
        ngrams = []

        for i in range(1, len(terms) + 1):  # number of words
            for start in range(0, len(terms) - i + 1):  # start point
                ngram = terms[start]
                quit = False

                if re.match(r'^[_\W]+$', terms[start]):
                    # Invalid input
                    continue

                for j in range(1, np.min([i, self.__n])):
                    # builds the sub-string
                    # If it is seperated by a trailing comma or whatever, then
                    # it is most likely an end of the sentence.
                    lookup = terms[start + j]
                    if not re.match(r'^[_\W]+$', lookup):
                        ngram += " {}".format(lookup)
                    else:
                        quit = True
                        break

                if quit:
                    continue

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

                end = self.__find_end_pos(ngram, query, pos)
                ngrams.append([ngram, pos, end])
        return ngrams