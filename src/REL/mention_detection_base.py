import os
import re
from functools import lru_cache

from REL.db.generic import GenericLookup
from REL.utils import modify_uppercase_phrase, split_in_words


class MentionDetectionBase:
    def __init__(self, base_url, wiki_version):
        self.wiki_db = GenericLookup(
            "entity_word_embedding", os.path.join(base_url, wiki_version, "generated")
        )

    def get_ctxt(self, start, end, idx_sent, sentence, sentences_doc):
        """
        Retrieves context surrounding a given mention up to 100 words from both sides.

        :return: left and right context
        """

        # Iteratively add words up until we have 100
        left_ctxt = split_in_words(sentence[:start])
        if idx_sent > 0:
            i = idx_sent - 1
            while (i >= 0) and (len(left_ctxt) <= 100):
                left_ctxt = split_in_words(sentences_doc[i]) + left_ctxt
                i -= 1
        left_ctxt = left_ctxt[-100:]
        left_ctxt = " ".join(left_ctxt)

        right_ctxt = split_in_words(sentence[end:])
        if idx_sent < len(sentences_doc):
            i = idx_sent + 1
            while (i < len(sentences_doc)) and (len(right_ctxt) <= 100):
                right_ctxt = right_ctxt + split_in_words(sentences_doc[i])
                i += 1
        right_ctxt = right_ctxt[:100]
        right_ctxt = " ".join(right_ctxt)

        return left_ctxt, right_ctxt

    @lru_cache(maxsize=None)
    def get_candidates(self, mention):
        """
        Retrieves a maximum of 100 candidates from the sqlite3 database for a given mention.

        :return: set of candidates
        """

        # Performs extra check for ED.
        cands = self.wiki_db.wiki(mention, "wiki")
        if cands:
            return cands[:100]
        else:
            return []

    @lru_cache(maxsize=None)
    def preprocess_mention(self, m):
        """
        Responsible for preprocessing a mention and making sure we find a set of matching candidates
        in our database.

        :return: mention
        """

        # TODO: This can be optimised (less db calls required).
        cur_m = modify_uppercase_phrase(m)
        freq_lookup_cur_m = self.wiki_db.wiki(cur_m, "wiki", "freq")

        if not freq_lookup_cur_m:
            cur_m = m

        freq_lookup_m = self.wiki_db.wiki(m, "wiki", "freq")
        freq_lookup_cur_m = self.wiki_db.wiki(cur_m, "wiki", "freq")

        if freq_lookup_m and (freq_lookup_m > freq_lookup_cur_m):
            # Cases like 'U.S.' are handed badly by modify_uppercase_phrase
            cur_m = m

        freq_lookup_cur_m = self.wiki_db.wiki(cur_m, "wiki", "freq")
        # If we cannot find the exact mention in our index, we try our luck to
        # find it in a case insensitive index.
        if not freq_lookup_cur_m:
            # cur_m and m both not found, verify if lower-case version can be found.
            find_lower = self.wiki_db.wiki(m.lower(), "wiki", "lower")

            if find_lower:
                cur_m = find_lower

        freq_lookup_cur_m = self.wiki_db.wiki(cur_m, "wiki", "freq")
        # Try and remove first or last characters (e.g. 'Washington,' to 'Washington')
        # To be error prone, we only try this if no match was found thus far, else
        # this might get in the way of 'U.S.' converting to 'US'.
        # Could do this recursively, interesting to explore in future work.
        if not freq_lookup_cur_m:
            temp = re.sub(r"[\(.|,|!|')]", "", m).strip()
            simple_lookup = self.wiki_db.wiki(temp, "wiki", "freq")

            if simple_lookup:
                cur_m = temp

        return cur_m
