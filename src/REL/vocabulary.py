import re

LOWER = False
DIGIT_0 = False
UNK_TOKEN = "#UNK#"

BRACKETS = {
    "-LCB-": "{",
    "-LRB-": "(",
    "-LSB-": "[",
    "-RCB-": "}",
    "-RRB-": ")",
    "-RSB-": "]",
}



class Vocabulary:
    """
    Class that creates a Vocabulary object that is used to store references to Embeddings.
    """
    unk_token = UNK_TOKEN

    def __init__(self):
        self.word2id = {}
        self.idtoword = {}

        self.id2word = []
        self.counts = []
        self.unk_id = 0
        self.first_run = 0

    @staticmethod
    def normalize(token, lower=LOWER, digit_0=DIGIT_0):
        """
        Normalises token.

        :return: Normalised token
        """

        if token in [Vocabulary.unk_token, "<s>", "</s>"]:
            return token
        elif token in BRACKETS:
            token = BRACKETS[token]
        else:
            if digit_0:
                token = re.sub("[0-9]", "0", token)

        if lower:
            return token.lower()
        else:
            return token

    def add_to_vocab(self, token):
        """
        Adds token to vocabulary.

        :return:
        """
        new_id = len(self.id2word)
        self.id2word.append(token)
        self.word2id[token] = new_id
        self.idtoword[new_id] = token

    def size(self):
        """
        Checks size vocabulary.

        :return: size vocabulary
        """
        return len(self.id2word)

    def get_id(self, token):
        """
        Normalises token and checks if token in vocab.

        :return: Either reference ID to given token or reference ID to #UNK# token.
        """
        tok = Vocabulary.normalize(token)
        return self.word2id.get(tok, self.unk_id)
