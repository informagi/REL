import json
import re
from functools import lru_cache

import flair
import numpy as np
import pkg_resources
from anyascii import anyascii
from colorama import Fore, Style
from flair.file_utils import get_from_cache
from nltk.tokenize import RegexpTokenizer


def fetch_model(path_or_url, cache_dir=flair.cache_root / "models/taggers"):
    model_dict = json.loads(pkg_resources.resource_string("REL.models", "models.json"))
    # load alias if it exists, else get original string
    path_or_url = model_dict.get(path_or_url, path_or_url)
    return get_from_cache(path_or_url, cache_dir)


def preprocess_mention(m, wiki_db):
    """
    Responsible for preprocessing a mention and making sure we find a set of matching candidates
    in our database.

    :return: mention
    """

    # TODO: This can be optimised (less db calls required).
    cur_m = modify_uppercase_phrase(m)
    freq_lookup_cur_m = wiki_db.wiki(cur_m, "wiki", "freq")

    if not freq_lookup_cur_m:
        cur_m = m

    freq_lookup_m = wiki_db.wiki(m, "wiki", "freq")
    freq_lookup_cur_m = wiki_db.wiki(cur_m, "wiki", "freq")

    if freq_lookup_m and (freq_lookup_m > freq_lookup_cur_m):
        # Cases like 'U.S.' are handed badly by modify_uppercase_phrase
        cur_m = m

    freq_lookup_cur_m = wiki_db.wiki(cur_m, "wiki", "freq")
    # If we cannot find the exact mention in our index, we try our luck to
    # find it in a case insensitive index.
    if not freq_lookup_cur_m:
        # cur_m and m both not found, verify if lower-case version can be found.
        find_lower = wiki_db.wiki(m.lower(), "wiki", "lower")

        if find_lower:
            cur_m = find_lower

    freq_lookup_cur_m = wiki_db.wiki(cur_m, "wiki", "freq")
    # Try and remove first or last characters (e.g. 'Washington,' to 'Washington')
    # To be error prone, we only try this if no match was found thus far, else
    # this might get in the way of 'U.S.' converting to 'US'.
    # Could do this recursively, interesting to explore in future work.
    if not freq_lookup_cur_m:
        temp = re.sub(r"[\(.|,|!|')]", "", m).strip()
        simple_lookup = wiki_db.wiki(temp, "wiki", "freq")

        if simple_lookup:
            cur_m = temp

    return cur_m


def process_results(
    mentions_dataset,
    predictions,
    processed,
    include_offset=False,
):
    """
    Function that can be used to process the End-to-End results.
    :return: dictionary with results and document as key.
    """
    res = {}
    for doc in mentions_dataset:
        if doc not in predictions:
            # No mentions found, we return empty list.
            continue
        pred_doc = predictions[doc]
        ment_doc = mentions_dataset[doc]
        text = processed[doc][0]
        res_doc = []

        for pred, ment in zip(pred_doc, ment_doc):
            sent = ment["sentence"]
            idx = ment["sent_idx"]
            start_pos = ment["pos"]
            mention_length = int(ment["end_pos"] - ment["pos"])

            if pred["prediction"] != "NIL":
                temp = (
                    start_pos,
                    mention_length,
                    ment["ngram"],
                    pred["prediction"],
                    pred["conf_ed"],
                    ment["conf_md"] if "conf_md" in ment else 0.0,
                    ment["tag"] if "tag" in ment else "NULL",
                )
                res_doc.append(temp)
        res[doc] = res_doc
    return res


def trim1(s):
    return s.replace("^%s*(.-)%s*$", "%1")


def first_letter_to_uppercase(s):
    if len(s) < 1:
        return s
    if len(s) == 1:
        return s.upper()
    return s[0].upper() + s[1:]


def modify_uppercase_phrase(s):
    if s == s.upper():
        return s.title()
    else:
        return s


@lru_cache(maxsize=None)
def anyascii_cached(t):
    return anyascii(t)


def split_in_words(inputstr):
    """
    This regexp also splits 'AL-NAHAR', which should be a single word
    into 'AL' and 'NAHAR', resulting in the inability to find a match.

    Same with U.S.
    """
    tokenizer = RegexpTokenizer(r"\w+")
    return [anyascii_cached(w) for w in tokenizer.tokenize(inputstr)]  # #inputstr.split()]#


def split_in_words_mention(inputstr):
    """
    This regexp also splits 'AL-NAHAR', which should be a single word
    into 'AL' and 'NAHAR', resulting in the inability to find a match.

    Same with U.S.
    """
    return [anyascii_cached(w) for w in inputstr.split()]  # #inputstr.split()]#


def correct_type(args, data):
    if "cuda" in args.type:
        return data.cuda()
    else:
        return data.cpu()


def flatten_list_of_lists(list_of_lists):
    """
    making inputs to torch.nn.EmbeddingBag
    """
    list_of_lists = [[]] + list_of_lists
    offsets = np.cumsum([len(x) for x in list_of_lists])[:-1]
    flatten = sum(list_of_lists[1:], [])
    return flatten, offsets


def make_equal_len(lists, fill_in=0, to_right=True):
    lens = [len(l) for l in lists]
    max_len = max(1, max(lens))
    if to_right:
        eq_lists = [l + [fill_in] * (max_len - len(l)) for l in lists]
        mask = [[1.0] * l + [0.0] * (max_len - l) for l in lens]
    else:
        eq_lists = [[fill_in] * (max_len - len(l)) + l for l in lists]
        mask = [[0.0] * (max_len - l) + [1.0] * l for l in lens]
    return eq_lists, mask


@lru_cache(maxsize=None)
def is_important_word(s):
    """
    an important word is not a stopword, a number, or len == 1
    """
    try:
        if len(s) <= 1 or s.lower() in STOPWORDS:
            return False
        float(s)
        return False
    except:
        return True


def is_stopword(s):
    return s.lower() in STOPWORDS


def tokgreen(s):
    print(f"{Fore.GREEN}{s}{Style.RESET_ALL}")


def tokfail(s):
    print(f"{Fore.RED}{s}{Style.RESET_ALL}")


def tokblue(s):
    print(f"{Fore.BLUE}{s}{Style.RESET_ALL}")


def unicode2ascii(c):
    return c.encode("ascii").decode("unicode-escape")


STOPWORDS = {
    "a",
    "about",
    "above",
    "across",
    "after",
    "afterwards",
    "again",
    "against",
    "all",
    "almost",
    "alone",
    "along",
    "already",
    "also",
    "although",
    "always",
    "am",
    "among",
    "amongst",
    "amoungst",
    "amount",
    "an",
    "and",
    "another",
    "any",
    "anyhow",
    "anyone",
    "anything",
    "anyway",
    "anywhere",
    "are",
    "around",
    "as",
    "at",
    "back",
    "be",
    "became",
    "because",
    "become",
    "becomes",
    "becoming",
    "been",
    "before",
    "beforehand",
    "behind",
    "being",
    "below",
    "beside",
    "besides",
    "between",
    "beyond",
    "both",
    "bottom",
    "but",
    "by",
    "call",
    "can",
    "cannot",
    "cant",
    "dont",
    "co",
    "con",
    "could",
    "couldnt",
    "cry",
    "de",
    "describe",
    "detail",
    "do",
    "done",
    "down",
    "due",
    "during",
    "each",
    "eg",
    "eight",
    "either",
    "eleven",
    "else",
    "elsewhere",
    "empty",
    "enough",
    "etc",
    "even",
    "ever",
    "every",
    "everyone",
    "everything",
    "everywhere",
    "except",
    "few",
    "fifteen",
    "fify",
    "fill",
    "find",
    "fire",
    "first",
    "five",
    "for",
    "former",
    "formerly",
    "forty",
    "found",
    "four",
    "from",
    "front",
    "full",
    "further",
    "get",
    "give",
    "go",
    "had",
    "has",
    "hasnt",
    "have",
    "he",
    "hence",
    "her",
    "here",
    "hereafter",
    "hereby",
    "herein",
    "hereupon",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "however",
    "hundred",
    "i",
    "ie",
    "if",
    "in",
    "inc",
    "indeed",
    "interest",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "keep",
    "last",
    "latter",
    "latterly",
    "least",
    "less",
    "ltd",
    "made",
    "many",
    "may",
    "me",
    "meanwhile",
    "might",
    "mill",
    "mine",
    "more",
    "moreover",
    "most",
    "mostly",
    "move",
    "much",
    "must",
    "my",
    "myself",
    "name",
    "namely",
    "neither",
    "never",
    "nevertheless",
    "next",
    "nine",
    "no",
    "nobody",
    "none",
    "noone",
    "nor",
    "not",
    "nothing",
    "now",
    "nowhere",
    "of",
    "off",
    "often",
    "on",
    "once",
    "one",
    "only",
    "onto",
    "or",
    "other",
    "others",
    "otherwise",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "part",
    "per",
    "perhaps",
    "please",
    "put",
    "rather",
    "re",
    "same",
    "see",
    "seem",
    "seemed",
    "seeming",
    "seems",
    "serious",
    "several",
    "she",
    "should",
    "show",
    "side",
    "since",
    "sincere",
    "six",
    "sixty",
    "so",
    "some",
    "somehow",
    "someone",
    "something",
    "sometime",
    "sometimes",
    "somewhere",
    "still",
    "such",
    "system",
    "take",
    "ten",
    "than",
    "that",
    "the",
    "their",
    "them",
    "themselves",
    "then",
    "thence",
    "there",
    "thereafter",
    "thereby",
    "therefore",
    "therein",
    "thereupon",
    "these",
    "they",
    "thick",
    "thin",
    "third",
    "this",
    "those",
    "though",
    "three",
    "through",
    "throughout",
    "thru",
    "thus",
    "to",
    "together",
    "too",
    "top",
    "toward",
    "towards",
    "twelve",
    "twenty",
    "two",
    "un",
    "under",
    "until",
    "up",
    "upon",
    "us",
    "very",
    "via",
    "was",
    "we",
    "well",
    "were",
    "what",
    "whatever",
    "when",
    "whence",
    "whenever",
    "where",
    "whereafter",
    "whereas",
    "whereby",
    "wherein",
    "whereupon",
    "wherever",
    "whether",
    "which",
    "while",
    "whither",
    "who",
    "whoever",
    "whole",
    "whom",
    "whose",
    "why",
    "will",
    "with",
    "within",
    "without",
    "would",
    "yet",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "st",
    "years",
    "yourselves",
    "new",
    "used",
    "known",
    "year",
    "later",
    "including",
    "used",
    "end",
    "did",
    "just",
    "best",
    "using",
}
