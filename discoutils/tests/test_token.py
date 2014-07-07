from discoutils.tokens import Token, DocumentFeature

__author__ = 'mmb28'


def test_tokens_ordering():
    cat = Token('cat', 'N', ner='O')
    cat_again = Token('cat', 'N', ner='O')
    dog = Token('dog', 'N', ner='O')
    dog_v = Token('dog', 'V', ner='O')

    assert cat < dog < dog_v
    assert dog != dog_v

    assert cat == cat_again


def test_smart_lower():
    # test that the PoS of an n-gram entry is not lowercased
    assert DocumentFeature.smart_lower('Cat/N') == 'cat/N'
    assert DocumentFeature.smart_lower('Cat/n') == 'cat/n'
    assert DocumentFeature.smart_lower('Red/J_CaT/N') == 'red/J_cat/N'
    assert DocumentFeature.smart_lower('Red/J CaT/N', separator=' ') == 'red/J cat/N'
    # test that features are not touched
    assert DocumentFeature.smart_lower('amod-DEP:former', lowercasing=False) == 'amod-DEP:former'