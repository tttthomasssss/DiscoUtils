__author__ = 'miroslavbatchkarov'
from discoutils.find_all_NPs import go, go_get_vectors
from cStringIO import StringIO


def test_find_all_NPs():
    s = StringIO()
    go('discoutils/tests/resources/exp10head.pbfiltered', s)
    expected = "AN:full-time/J_tribunal/N\n" \
               "AN:ordinary/J_session/N\n" \
               "NN:council/N_session/N\n" \
               "AN:large/J_cat/N\n" \
               "AN:fluffy/J_cat/N\n" \
               "NN:house/N_cat/N\n" \
               "NN:street/N_cat/N\n" \
               "AN:troubled/J_activist/N\n" \
               "AN:leftw/J_activist/N\n"
    assert s.getvalue() == expected


def test_find_all_NPs_with_seed():
    s = StringIO()
    go('discoutils/tests/resources/exp10head.pbfiltered', s, seed_set={'ordinary/J'})
    expected = "AN:ordinary/J_session/N\n"
    assert s.getvalue() == expected


def test_find_all_NPs_with_window_vectors():
    s = StringIO()
    go_get_vectors('discoutils/tests/resources/exp10head.pbfiltered', s)
    expected_features = {'T:something/N', 'T:large/J', 'T:what/CONJ', 'T:ever/N', 'T:feature/J'}
    lines = s.getvalue().rstrip().split('\n')
    from pprint import pprint

    pprint(lines)
    assert len(lines) == 7  # only some NPs have window features. Some lines in the test
    # file mention multiple NPs- all need to be produced.
    # Just check that the right features are returned, by do not bother with checking their ordering
    assert expected_features == set(feature for line in lines for feature in line.split('\t')[1:])
