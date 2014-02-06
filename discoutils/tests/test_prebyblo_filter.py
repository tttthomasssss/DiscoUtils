import pytest
import re
from prebyblo_filter import count, do_filtering


@pytest.fixture
def filename():
    return 'discoutils/tests/resources/exp6head'


def test_count(filename):
    counts, total = count(filename,
                          [re.compile('.*/N'), re.compile('.*/V'),
                           re.compile('.*/J'), re.compile('.*/RB')])
    assert counts == {'be/V': 3, 'blame/V': 1, 'bad/J': 1, 'scare/V': 1, 'budgetary/J': 1,
                      'construction/N': 1, 'consideration/N': 1, 'get/V': 1, 'have/V': 1}
    assert total == 22

    counts, total = count(filename, [re.compile('.*/N')])
    assert counts == {'construction/N': 1, 'consideration/N': 1}
    assert total == 22


def test_do_filtering(filename):
    from cStringIO import StringIO

    s = StringIO()
    counts, total = count('discoutils/tests/resources/exp6head', [re.compile('.*/N')])
    do_filtering(filename, s, 0, [re.compile('.*/N')], re.compile('T:.*'), counts, total)

    expected = "consideration/N	T:``	T:there	T:have	T:to	T:be	T:budgetary	T:.\n" \
               "construction/N	T:it	T:be	T:blame	T:on	T:bad	T:.\n"
    assert s.getvalue() == expected

    s = StringIO()
    do_filtering(filename, s, 0, [re.compile('.*/N')], re.compile('.+-(HEAD|DEP):.+'), counts, total)

    expected = "consideration/N	amod-DEP:budgetary	cop-DEP:be	xcomp-HEAD:have	aux-DEP:to\n" \
               "construction/N	amod-DEP:bad	pobj-HEAD:on\n"
    assert s.getvalue() == expected