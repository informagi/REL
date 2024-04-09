class Token:
    def __init__(self, text, start_pos, end_pos, score, tag):
        self.text = text
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.score = score
        self.tag = tag


class MD_Module(object):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def predict(self, sentence, sentences_doc):
        """
        The module takes as an input a sentence from the current doc and all the sentences of
        the current doc (for global context). The user is expected to return a list of mentions,
        where each mention is a Token class.

        We denote the following requirements:
        1. Any MD module should have a 'predict()' function that returns a list of mentions.
        2. A mention is always defined as a Token class (see above).

        """
        # returns list of Token objects.
        return self.find_mentions()

    def find_mentions(self, sentence):
        mentions = []
        for i in range(10):
            mentions.append(Token(i, i, i, i, i))
        return mentions
