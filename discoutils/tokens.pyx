from functools import total_ordering
from itertools import zip_longest
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
    pos_separator = '/'
    ngram_separator = '_'
    BANNED_CHARS = set('*.+="?\|%012356789')

    @classmethod
    def recompile_pattern(cls, pos_separator='/', ngram_separator='_'):
        """
        If you want to use non-standard separators, call this method first
        :param pos_separator:
        :param ngram_separator:
        """
        #  not an underscore + text + underscore or end of line
        #  see re.split documentation on capturing (the first two) and non-capturing groups (the last one)
        cls.pos_separator = pos_separator
        cls.ngram_separator = ngram_separator

    @classmethod
    def from_string(cls, string):
        """
        Parses a string and creates a document feature out of it. This is the inverse of
        this class' __str__. Returns an EMPTY feature on exception
        :param string: the string to parse
        :return: DocumentFeature corresponding to the input string or DocumentFeature(type, tuple(tokens))
        """
        if isinstance(string, DocumentFeature):
            return string

        try:
            token_strings = string.split(cls.ngram_separator)
            tokens = []
            for t in token_strings:
                wordpos = t.split(cls.pos_separator)
                if len(wordpos) == 1:
                    word, pos = wordpos[0], None
                elif len(wordpos) == 2:
                    word, pos = wordpos
                else:
                    word, pos = None, None
                if (not word) or set(word).intersection(cls.BANNED_CHARS):
                    raise ValueError

                tokens.append(Token(word, pos, pos_separator=cls.pos_separator))

            pos_tags = [t.pos for t in tokens if t.pos]
            if 0 < len(pos_tags) < len(tokens):
                raise ValueError
            pos_sequence = ''.join(t.upper() for t in pos_tags)
            feat_type = cls._TYPES.get(pos_sequence, ('EMPTY', '1-GRAM', '2-GRAM', '3-GRAM')[len(tokens)])
            return DocumentFeature(feat_type, tuple(tokens))
        except Exception as e:
            return DocumentFeature('EMPTY', tuple())

    @classmethod
    def smart_lower(cls, words_with_pos, lowercasing=True):
        """
        Lowercase just the words and not their PoS tags
        """
        if not lowercasing:
            return words_with_pos

        unigrams = words_with_pos.split(cls.ngram_separator)
        words = []
        for unigram in unigrams:
            try:
                word, pos = unigram.split(cls.pos_separator)
            except ValueError:
                # no pos
                word, pos = words_with_pos, ''

            words.append(cls.pos_separator.join([word.lower(), pos]) if pos else word.lower())

        return cls.ngram_separator.join(words)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item):
        '''
        Override slicing operator. Creates a feature from the tokens of this feature between
        positions beg (inclusive) and end (exclusive). For example:

        >>> f = DocumentFeature.from_string('cats/N_like/V_dogs/N')
        >>> print(f[1])
        like/V
        >>> print(f[1:])
        like/V_dogs/N
        >>> print(f[0:])
        cats/N_like/V_dogs/N
        >>> print(f[0])
        cats/N


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
        return self.ngram_separator.join(str(t) for t in self.tokens)

    def __repr__(self):
        return 'DF:' + str(self)

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
    PoS tag and NER tag.

    The index field may ignored by __hash__, __eq__ and the like. This is because it is irrelevant
    when checking if a token is contained in a sentence- most of the time where exactly is not
    important. However, the index matters when building a dependency tree of looking up the
    neighbours of a given token.

    """

    def __init__(self, text, pos, index='any', ner='O', pos_separator='/', **kwargs):
        self.text = text
        self.pos = pos
        self.index = index  # useful when parsing CONLL
        self.ner = ner
        self.pos_separator = pos_separator

    def __str__(self):
        return '{}{}{}'.format(self.text, self.pos_separator, self.pos) if self.pos else self.text

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return (not self < other) and (not other < self)

    def __lt__(self, other):
        if self.index == 'any' or other.index == 'any':
            return self.text < other.text
        else:
            return (self.text, self.index) < (other.text, other.index)

    def __hash__(self):
        return hash((self.text, self.pos))
