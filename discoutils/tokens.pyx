from functools import total_ordering
import logging
from operator import itemgetter
from itertools import izip_longest
import re


class DocumentFeature(object):
    """
    Represents an n-gram document feature. Each feature has a syntactic type, currently
        SVO- subject-verb-object
        VO- verbo-object
        AN- adjective-nouns
        NN- noun-noun
        1-GRAM, 2-GRAM, 3-GRAM
        EMPTY- if an attempt to parse an invalid string has been made
    """
    def __init__(self, type, tokens):
        """

        :param type: the type of the feature, string
        :param tokens: a tuple of Token objects that this DocumentFeature consists of
        """
        self.type = type
        self.tokens = tokens

    _TYPES = dict([('NVN', 'SVO'), ('JN', 'AN'), ('VN', 'VO'), ('NN', 'NN')])
    #  not an underscore + text + underscore or end of line
    #  see re.split documentation on capturing (the first two) and non-capturing groups (the last one)
    _TOKEN_RE = re.compile(r'([^/_]+)/([A-Z]+)(?:_|$)')

    @classmethod
    def from_string(cls, string):
        """
        Parses a string and creates a document feature out of it. This is the inverse of
        this class' tokens_as_str(). Returns an EMPTY feature on exception
        :param string: the string to parse
        :return: DocumentFeature corresponding to the input string or DocumentFeature(type, tuple(tokens))
        """
        try:
            match = cls._TOKEN_RE.split(string, 3)
            type = ''.join(match[2::3])
            match = iter(match)
            tokens = []
            for (junk, word, pos) in izip_longest(match, match, match):
                if junk:        # Either too many tokens, or invalid token
                    raise ValueError(junk)
                if not word:
                    break
                tokens.append(Token(word, pos))
            type = cls._TYPES.get(type,
                                  ('EMPTY', '1-GRAM', '2-GRAM', '3-GRAM')[len(tokens)])
            return DocumentFeature(type, tuple(tokens))
        except:
            logging.error('Cannot create token out of string %s', string)
            return DocumentFeature('EMPTY', tuple())

    def tokens_as_str(self):
        """
        Represents the features of this document as a human-readable string
        DocumentFeature('1-GRAM', ('X', 'Y',)) -> 'X_Y'
        """
        return '_'.join(str(t) for t in self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item):
        '''
        Override slicing operator. Creates a feature from the tokens of this feature between
        positions beg (inclusive) and end (exclusive). For example:

        >>> f = DocumentFeature.from_string('cats/N_like/V_dogs/N')
        >>> print f[1]
        1-GRAM:(like/V,)
        >>> print f[1:]
        VO:(like/V, dogs/N)
        >>> print f[0:]
        SVO:(cats/N, like/V, dogs/N)
        >>> print f[0]
        1-GRAM:(cats/N,)


        :param beg:
        :type beg: int
        :param end:
        :type end: int or None
        :return:
        :rtype: DocumentFeature
        '''
        tokens = self.tokens[item]
        try:
            l = len(tokens)
            return DocumentFeature.from_string('_'.join(map(str, tokens)))
        except TypeError:
            # a single token has no len
            return DocumentFeature.from_string(str(tokens))

    def __str__(self):
        """
        Returns a human-readable representation of this object, including type information, e.g.
            DocumentFeature('1-GRAM', ('X', 'Y',)) -> '1-GRAM:X_Y'
        """
        return '{}:{}'.format(self.type, self.tokens)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self.__dict__ == other.__dict__)

    def __lt__(self, other):
        return (self.type, self.tokens) < (other.type, other.tokens)

    def __hash__(self):
        return hash((self.type, self.tokens))


@total_ordering
class Token(object):
    """
    Represents a text token. Stores information about the text, PoS/NER tag and index in a sentence.
    Tokens are printed with their PoS tag, e.g. cat/N, and ordered alphabetically by text,
    PoS tag and NER tag. The index field is ignored by __hash__, __eq__ and the like

    """
    def __init__(self, text, pos, index=0, ner='O'):
        self.text = text
        self.pos = pos
        self.index = index # useful when parsing CONLL
        self.ner = ner

    def __str__(self):
        return '{}/{}'.format(self.text, self.pos) if self.pos else self.text

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return (not self < other) and (not other < self)

    def __lt__(self, other):
        return (self.text, self.pos, self.ner) < (other.text, other.pos, other.ner)

    def __hash__(self):
        return hash((self.text, self.pos, self.ner))