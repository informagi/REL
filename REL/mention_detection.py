from REL.db.generic import GenericLookup
from REL.utils import preprocess_mention, split_in_words
from flair.data import Sentence
from flair.models import SequenceTagger
from segtok.segmenter import split_single


class MentionDetection:
    """
    Class responsible for mention detection.
    """

    def __init__(self, base_url, wiki_subfolder):
        self.cnt_exact = 0
        self.cnt_partial = 0
        self.cnt_total = 0
        self.wiki_db = GenericLookup(
            "entity_word_embedding",
            "{}/{}/generated/".format(base_url, wiki_subfolder),
        )

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
                left_ctxt = split_in_words(self.__sentences_doc[i]) + left_ctxt
                i -= 1
        left_ctxt = left_ctxt[-100:]
        left_ctxt = " ".join(left_ctxt)

        right_ctxt = split_in_words(sentence[end:])
        if idx_sent < len(self.__sentences_doc):
            i = idx_sent + 1
            while (i < len(self.__sentences_doc)) and (len(right_ctxt) <= 100):
                right_ctxt = right_ctxt + split_in_words(self.__sentences_doc[i])
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

        dataset, _, _ = self.split_text(dataset)
        results = {}
        total_ment = 0

        for doc in dataset:
            contents = dataset[doc]
            self.__sentences_doc = [v[0] for v in contents.values()]

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

    def split_text(self, dataset, is_flair=False):
        """
        Splits text into sentences with optional spans (format is a requirement for GERBIL usage).
        This behavior is required for the default NER-tagger, which during experiments was experienced
        to achieve higher performance.

        :return: dictionary with sentences and optional given spans per sentence.
        """

        res = {}
        splits = [0]
        processed_sentences = []
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
                if len(spans) == 0:
                    processed_sentences.append(
                        Sentence(sent, use_tokenizer=True) if is_flair else sent
                    )
                i += 1
            splits.append(splits[-1] + i)
        return res, processed_sentences, splits

    def find_mentions(self, dataset, tagger):
        """
        Responsible for finding mentions given a set of documents in a batch-wise manner. More specifically,
        it returns the mention, its left/right context and a set of candidates.

        :return: Dictionary with mentions per document.
        """

        if tagger is None:
            raise Exception(
                "No NER tagger is set, but you are attempting to perform Mention Detection.."
            )

        # Verify if Flair, else ngram or custom.
        is_flair = isinstance(tagger, SequenceTagger)
        dataset, processed_sentences, splits = self.split_text(dataset, is_flair)
        results = {}
        total_ment = 0

        # mini_batch_size default 32. Only if Flair for higher performance (GPU),
        # else predict on a sentence-level.
        if is_flair:
            tagger.predict(processed_sentences)

        for i, doc in enumerate(dataset):
            contents = dataset[doc]
            self.__sentences_doc = [v[0] for v in contents.values()]
            sentences = processed_sentences[splits[i] : splits[i + 1]]
            result_doc = []

            for (idx_sent, (sentence, ground_truth_sentence)), snt in zip(
                contents.items(), sentences
            ):
                for entity in (
                    snt.get_spans("ner")
                    if is_flair
                    else tagger.predict(snt, processed_sentences)
                ):
                    text, start_pos, end_pos, conf, tag = (
                        entity.text,
                        entity.start_pos,
                        entity.end_pos,
                        entity.score,
                        entity.tag,
                    )
                    total_ment += 1

                    m = preprocess_mention(text, self.wiki_db)
                    cands = self._get_candidates(m)

                    if len(cands) == 0:
                        continue

                    # Re-create ngram as 'text' is at times changed by Flair (e.g. double spaces are removed).
                    ngram = sentence[start_pos:end_pos]
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
                        "tag": tag,
                    }

                    result_doc.append(res)

            results[doc] = result_doc

        return results, total_ment
