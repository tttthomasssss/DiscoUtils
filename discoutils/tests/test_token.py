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
    assert DocumentFeature.smart_lower('Cat/n') == 'cat/n'
    assert DocumentFeature.smart_lower('Red/J_CaT/N') == 'red/J_cat/N'
    # test that features are not touched
    assert DocumentFeature.smart_lower('amod-DEP:former', lowercasing=False) == 'amod-DEP:former'

    DocumentFeature.recompile_pattern(ngram_separator=' ')
    assert DocumentFeature.smart_lower('Red/J CaT/N') == 'red/J cat/N'

    DocumentFeature.recompile_pattern(pos_separator='-')
    assert DocumentFeature.smart_lower('Red-J') == 'red-J'


def test_document_feature_from_string():
    DocumentFeature.recompile_pattern()
    x = DocumentFeature.from_string('big/J_cat/N')
    y = DocumentFeature('AN', (Token('big', 'J'), Token('cat', 'N')))
    assert y == x

    assert DocumentFeature('1-GRAM', (Token('cat', 'N'), )) == DocumentFeature.from_string('cat/N')

    assert DocumentFeature('VO', (Token('chase', 'V'), Token('cat', 'N'))) == \
           DocumentFeature.from_string('chase/V_cat/N')

    assert DocumentFeature('NN', (Token('dog', 'N'), Token('cat', 'N'))) == \
           DocumentFeature.from_string('dog/N_cat/N')

    assert DocumentFeature('NN', (Token('dog', 'N'), Token('cat', 'N'))) == \
           DocumentFeature.from_string('dog/n_cat/n')

    assert DocumentFeature('3-GRAM', (Token('dog', 'V'), Token('chase', 'V'), Token('cat', 'V'))) == \
           DocumentFeature.from_string('dog/V_chase/V_cat/V')

    assert DocumentFeature('2-GRAM', (Token('chase', 'V'), Token('cat', 'V'))) == \
           DocumentFeature.from_string('chase/V_cat/V')

    assert DocumentFeature('SVO', (Token('dog', 'N'), Token('chase', 'V'), Token('cat', 'N'))) == \
           DocumentFeature.from_string('dog/N_chase/V_cat/N')

    assert DocumentFeature('2-GRAM', (Token('very', 'RB'), Token('big', 'J'))) == \
           DocumentFeature.from_string('very/RB_big/J')

    for invalid_string in ['a\/s/N', 'l\/h/N_clinton\/south/N', 'l\/h//N_clinton\/south/N',
                           'l//fasdlj/fasd/dfs/sdf', 'l//fasdlj/fasd/dfs\_/sdf', 'dfs\_/sdf',
                           'dfs\_/fadslk_/sdf', '/_dfs\_/sdf', '_/_/', '_///f_/', 'drop_bomb',
                           'drop/V_bomb', '/V_/N', 'cat', 'word1_word2//', 'mk8/N_6hp/N',
                           'a./N_gordon/N', 'great/J_c.d./N', 'info@tourmate.com/N', 'w1/N',
                           '-lrb-306-rrb- 569-1995/N', 'mumaharps.com/N', 'c+l+a+v+i+e+r+/N',
                           'b/N_o\o/N', '%/N', '|/V']:
        print(invalid_string)
        assert DocumentFeature('EMPTY', tuple()) == DocumentFeature.from_string(invalid_string)


def test_document_feature_slicing():
    DocumentFeature.recompile_pattern()
    x = DocumentFeature.from_string('big/J_cat/N')
    assert x[0] == DocumentFeature.from_string('big/J')
    assert x[1] == DocumentFeature.from_string('cat/N')
    assert x[1] == DocumentFeature('1-GRAM', (Token('cat', 'N', 1), ))
    assert x[0:] == DocumentFeature.from_string('big/J_cat/N')

    x = DocumentFeature.from_string('cat/N')
    assert x[0] == DocumentFeature.from_string('cat/N')
    assert x[0:] == DocumentFeature.from_string('cat/N')
    assert x[:] == DocumentFeature.from_string('cat/N')


def test_with_different_separators():
    DocumentFeature.recompile_pattern(pos_separator='_', ngram_separator='!')
    assert DocumentFeature('2-GRAM', (Token('very', 'RB'), Token('big', 'J'))) == \
           DocumentFeature.from_string('very_RB!big_J')

    DocumentFeature.recompile_pattern(pos_separator='-', ngram_separator=' ')
    assert DocumentFeature('1-GRAM', (Token('very', 'RB'),)) == DocumentFeature.from_string('very-RB')
    assert DocumentFeature('2-GRAM', (Token('very', 'RB'), Token('big', 'J'))) == \
           DocumentFeature.from_string('very-RB big-J')