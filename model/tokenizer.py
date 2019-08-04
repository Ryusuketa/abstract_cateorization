from nltk import SpaceTokenizer
from typing import Dict

import luigi


class PreTrainedSpaceTokenizer(object):
    def __init__(self, token2id: Dict[str, int]):
        self.token2id = token2id
        self.SpaceTokenizer = SpaceTokenizer()

    def encode(self, sentence):
        tokens = self.SpaceTokenizer.tokenize(sentence)
        token_ids = [self.token2id[token] for token in tokens if token in self.token2id.keys()]

        return token_ids