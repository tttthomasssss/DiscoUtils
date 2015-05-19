from discoutils.find_all_NPs import (get_NPs, get_VPs, get_window_vectors_for_NPs, get_window_vectors_for_VPs)

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

expected_NPs = [
    'full-time/J_tribunal/N',
    'ordinary/J_session/N',
    'council/N_session/N',
    'large/J_cat/N',
    'fluffy/J_cat/N',
    'house/N_cat/N',
    'street/N_cat/N',
    'troubled/J_activist/N',
    'leftw/J_activist/N'
]
expected_VPs = ['launch/V_attack/N',
                'keep/V_site/N',
                'strabag/N_express/V_interest/N',
                'activist/N_tell/V_police/N',
                'karzai/N_declare/V_thursday/N']


def test_find_all_NPs():
    s = StringIO()
    get_NPs('discoutils/tests/resources/exp10head.pbfiltered', s)
    assert s.getvalue() == '\n'.join(expected_NPs) + '\n'


def test_find_all_NPs_with_seed():
    s = StringIO()
    get_NPs('discoutils/tests/resources/exp10head.pbfiltered', s, whitelist={'ordinary/J'})
    expected = 'ordinary/J_session/N\n'
    assert s.getvalue() == expected


def test_find_all_NPs_with_window_vectors():
    s = StringIO()
    get_window_vectors_for_NPs('discoutils/tests/resources/exp10head.pbfiltered', s)

    expected_features = {'T:problem/N', 'T:large/J',
                         # 'T:what/CONJ', 'T:ever/N', # these are in the file but are stopwords
                         'T:feature/J', 'T:cat/N'}
    lines = s.getvalue().rstrip().split('\n')
    for l in lines:
        print(l)
    assert len(lines) == len(expected_NPs)  # one line per entry occurence
    # only some NPs have window features. Some lines in the test
    # file mention multiple NPs- all need to be produced. Some NNs appear twice.
    # Just check that the right features are returned, by do not bother with checking their ordering
    assert expected_features == set(feature for line in lines for feature in line.split('\t')[1:])
    assert set(expected_NPs) == set(line.split('\t')[0] for line in lines)


def test_find_all_NPs_with_window_vectors_with_filtering():
    s = StringIO()
    get_window_vectors_for_NPs('discoutils/tests/resources/exp10head.pbfiltered', s,
                               whitelist={'council/N_session/N', 'large/J_cat/N'})
    expected_entries = ['council/N_session/N', 'large/J_cat/N']

    expected_features = {'T:feature/J', 'T:cat/N'}
    lines = s.getvalue().rstrip().split('\n')
    for l in lines:
        print(l)
    assert len(lines) == 2
    assert expected_features == set(feature for line in lines for feature in line.split('\t')[1:])
    assert expected_entries == [line.split('\t')[0] for line in lines]


def test_find_all_VPs():
    s = StringIO()
    get_VPs('discoutils/tests/resources/exp10head-2.pbfiltered', s)

    print(s.getvalue())
    assert s.getvalue() == '\n'.join(expected_VPs) + '\n'


def test_find_all_VPs_with_window_vectors():
    s = StringIO()
    get_window_vectors_for_VPs('discoutils/tests/resources/exp10head-2.pbfiltered', s)

    expected_features = set(['T:ordinary/J', 'T:council/N', 'T:session/N', 'T:preserve/V', 'T:include/V',
                             'T:building/N', 'T:builder/N',
                             # 'T:strabag/N', 'T:declare/V', 'T:site/N', # these are there but are part of the phrase
                             'T:life/N', 'T:source/N', 'T:mourning/N', 'T:state/N'])
    lines = s.getvalue().rstrip().split('\n')
    for l in lines:
        print(l)
    assert len(lines) == len(expected_VPs)
    actual_features = set((feature for line in lines for feature in line.split('\t')[1:]))
    print(expected_features - actual_features)
    assert expected_features == actual_features
    assert set(expected_VPs) == set(line.split('\t')[0] for line in lines)