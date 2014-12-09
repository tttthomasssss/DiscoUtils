from discoutils.tokens import Token, DocumentFeature

__author__ = 'mmb28'


def test_tokens_ordering():
    cat = Token('cat', 'N', ner='O')
    cat_again = Token('cat', 'N', ner='O')
    dog = Token('dog', 'N', ner='O')
    dog_v = Token('dog', 'V', ner='O')

    assert cat < dog == dog_v
    assert cat == cat_again


def test_token_to_string():
    assert 'dog/J' == str(DocumentFeature.from_string('dog/J').tokens[0])
    DocumentFeature.recompile_pattern(pos_separator='-')
    my_feature = DocumentFeature.from_string('dog-J')
    assert 'dog-J' == my_feature.tokens_as_str()
    DocumentFeature.recompile_pattern()


def test_smart_lower():
    # test that the PoS of an n-gram entry is not lowercased
    assert DocumentFeature.smart_lower('Cat/N') == 'cat/N'
    assert DocumentFeature.smart_lower('Cat/n') == 'cat/N'
    assert DocumentFeature.smart_lower('Red/J_CaT/N') == 'red/J_cat/N'
    # test that features are not touched
    assert DocumentFeature.smart_lower('amod-DEP:former', lowercasing=False) == 'amod-DEP:former'

    DocumentFeature.recompile_pattern(ngram_separator=' ')
    assert DocumentFeature.smart_lower('Red/J CaT/N') == 'red/J cat/N'

    DocumentFeature.recompile_pattern(pos_separator='-')
    assert DocumentFeature.smart_lower('Red-J') == 'red-J'
