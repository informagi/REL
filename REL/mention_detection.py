from REL.utils import preprocess_mention, split_in_words

from REL.db.generic import GenericLookup
from segtok.segmenter import split_single
from flair.data import Sentence

"""
Class responsible for mention detection. 
"""


class MentionDetection:
    def __init__(self, base_url, wiki_subfolder):
        self.cnt_exact = 0
        self.cnt_partial = 0
        self.cnt_total = 0
        self.wiki_db = GenericLookup(
            "entity_word_embedding",
            "{}/{}/generated/".format(base_url, wiki_subfolder),
        )

    # def __verify_pos(self, ngram, start, end, sentence):
    #     ngram = ngram.lower()
    #     find_ngram = sentence[start:end].lower()
    #     find_ngram_ws_invariant = " ".join(
    #         [x.text for x in Sentence(find_ngram, use_tokenizer=True)]
    #     ).lower()

    #     assert (find_ngram == ngram) or (
    #         find_ngram_ws_invariant == ngram
    #     ), "Mention not found on given position: {};{};{};{}".format(
    #         find_ngram, ngram, find_ngram_ws_invariant, sentence
    #     )

    def split_text(self, dataset):
        """
        Splits text into sentences. This behavior is required for the default NER-tagger, which during experiments
        was experienced to perform more optimally in such a fashion.

        :return: dictionary with sentences and optional given spans per sentence.
        """

        res = {}
        for doc in dataset:
            text, spans = dataset[doc]
            sentences = split_single(text)
            res[doc] = {}

            i = 0
            for sent in sentences:
                if len(sent.strip()) == 0:
                    continue
                # Match gt to sentence.
                pos_start = text.find(sent)
                pos_end = pos_start + len(sent)

                # ngram, start_pos, end_pos
                spans_sent = [
                    [text[x[0] : x[0] + x[1]], x[0], x[0] + x[1]]
                    for x in spans
                    if pos_start <= x[0] < pos_end
                ]
                res[doc][i] = [sent, spans_sent]
                i += 1
        return res

    def _get_ctxt(self, start, end, idx_sent, sentence):
        """
        Retrieves context surrounding a given mention up to 100 words from both sides.

        :return: left and right context
        """

        # Iteratively add words up until we have 100
        left_ctxt = split_in_words(sentence[:start])
        if idx_sent > 0:
            i = idx_sent - 1
            while (i >= 0) and (len(left_ctxt) <= 100):
                left_ctxt = split_in_words(self.sentences_doc[i]) + left_ctxt
                i -= 1
        left_ctxt = left_ctxt[-100:]
        left_ctxt = " ".join(left_ctxt)

        right_ctxt = split_in_words(sentence[end:])
        if idx_sent < len(self.sentences_doc):
            i = idx_sent + 1
            while (i < len(self.sentences_doc)) and (len(right_ctxt) <= 100):
                right_ctxt = right_ctxt + split_in_words(self.sentences_doc[i])
                i += 1
        right_ctxt = right_ctxt[:100]
        right_ctxt = " ".join(right_ctxt)

        return left_ctxt, right_ctxt

    def _get_candidates(self, mention):
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

    def format_spans(self, dataset):
        """
        Responsible for formatting given spans into dataset for the ED step. More specifically,
        it returns the mention, its left/right context and a set of candidates.

        :return: Dictionary with mentions per document.
        """

        dataset = self.split_text(dataset)
        results = {}
        total_ment = 0

        for doc in dataset:
            contents = dataset[doc]
            self.sentences_doc = [v[0] for v in contents.values()]

            results_doc = []
            for idx_sent, (sentence, spans) in contents.items():
                for ngram, start_pos, end_pos in spans:
                    total_ment += 1

                    # end_pos = start_pos + length
                    # ngram = text[start_pos:end_pos]
                    mention = preprocess_mention(ngram, self.wiki_db)
                    left_ctxt, right_ctxt = self._get_ctxt(
                        start_pos, end_pos, idx_sent, sentence
                    )

                    chosen_cands = self._get_candidates(mention)
                    res = {
                        "mention": mention,
                        "context": (left_ctxt, right_ctxt),
                        "candidates": chosen_cands,
                        "gold": ["NONE"],
                        "pos": start_pos,
                        "sent_idx": idx_sent,
                        "ngram": ngram,
                        "end_pos": end_pos,
                        "sentence": sentence,
                    }

                    results_doc.append(res)
            results[doc] = results_doc
        return results, total_ment

    def find_mentions(self, dataset, tagger_ner=None):
        """
        Responsible for finding mentions given a set of documents. More specifically,
        it returns the mention, its left/right context and a set of candidates.

        :return: Dictionary with mentions per document.
        """

        if tagger_ner is None:
            raise Exception(
                "No NER tagger is set, but you are attempting to perform Mention Detection.."
            )

        dataset = self.split_text(dataset)
        results = {}
        total_ment = 0

        for doc in dataset:
            contents = dataset[doc]

            self.sentences_doc = [v[0] for v in contents.values()]
            result_doc = []

            sentences = [
                Sentence(v[0], use_tokenizer=True) for k, v in contents.items()
            ]

            tagger_ner.predict(sentences)

            for (idx_sent, (sentence, ground_truth_sentence)), snt in zip(
                contents.items(), sentences
            ):
                illegal = []
                for entity in snt.get_spans("ner"):
                    text, start_pos, end_pos, conf = (
                        entity.text,
                        entity.start_pos,
                        entity.end_pos,
                        entity.score,
                    )
                    total_ment += 1

                    m = preprocess_mention(text, self.wiki_db)
                    cands = self._get_candidates(m)

                    if len(cands) == 0:
                        continue

                    ngram = sentence[start_pos:end_pos]
                    illegal.extend(range(start_pos, end_pos))

                    left_ctxt, right_ctxt = self._get_ctxt(
                        start_pos, end_pos, idx_sent, sentence
                    )

                    res = {
                        "mention": m,
                        "context": (left_ctxt, right_ctxt),
                        "candidates": cands,
                        "gold": ["NONE"],
                        "pos": start_pos,
                        "sent_idx": idx_sent,
                        "ngram": ngram,
                        "end_pos": end_pos,
                        "sentence": sentence,
                        "conf_md": conf,
                        "tag": entity.tag,
                    }

                    result_doc.append(res)

            results[doc] = result_doc

        return results, total_ment
