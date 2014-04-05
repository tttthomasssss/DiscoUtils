from discoutils.tokens import Token

__author__ = 'mmb28'


def test_tokens_ordering():
    cat = Token('cat', 'N', ner='O')
    cat_again = Token('cat', 'N', ner='O')
    dog = Token('dog', 'N', ner='O')
    dog_v = Token('dog', 'V', ner='O')

    assert cat < dog < dog_v
    assert dog != dog_v

    assert cat == cat_again