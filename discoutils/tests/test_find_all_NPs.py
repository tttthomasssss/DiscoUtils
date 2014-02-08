__author__ = 'miroslavbatchkarov'
from discoutils.find_all_NPs import go
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
               "AN:troubled/J_activist/N\n"
    print '\n', s.getvalue()
    assert s.getvalue() == expected