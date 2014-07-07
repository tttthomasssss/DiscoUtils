# coding=utf-8
from bsddb3.db import DBAccessError
import shelve
from unittest import TestCase
import os

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from discoutils.thesaurus_loader import Thesaurus, Vectors
from discoutils.collections_utils import walk_nonoverlapping_pairs


__author__ = 'mmb28'


@pytest.fixture
def thesaurus_c():
    return Thesaurus.from_tsv(tsv_files=['discoutils/tests/resources/exp0-0c.strings'],
                              sim_threshold=0,
                              include_self=False,
                              ngram_separator='_')


@pytest.fixture
def vectors_c():
    return Vectors.from_tsv(tsv_files=['discoutils/tests/resources/exp0-0c.strings'],
                            sim_threshold=0,
                            ngram_separator='_')


@pytest.fixture
def thes_without_overlap():
    return Thesaurus.from_tsv(tsv_files=['discoutils/tests/resources/lexical-overlap.txt'],
                              sim_threshold=0,
                              ngram_separator='_',
                              allow_lexical_overlap=False)


@pytest.fixture
def thes_with_overlap():
    return Thesaurus.from_tsv(tsv_files=['discoutils/tests/resources/lexical-overlap.txt'],
                              sim_threshold=0,
                              ngram_separator='_',
                              allow_lexical_overlap=True)


def test_loading_bigram_thesaurus(thesaurus_c):
    assert len(thesaurus_c) == 5
    assert 'a/J_b/N' in thesaurus_c.keys()
    assert 'messed_up' not in thesaurus_c.keys()


def test_disallow_lexical_overlap(thes_without_overlap):
    # entries wil only overlapping neighbours must be removed
    assert len(thes_without_overlap) == 3
    # check the right number of neighbours are kept
    assert len(thes_without_overlap['monetary/J_screw/N']) == 1
    assert len(thes_without_overlap['daily/J_pais/N']) == 2
    # check the right neighbour is kept
    assert thes_without_overlap['japanese/J_yen/N'][0] == ('daily/J_mark/N', 0.981391)


@pytest.mark.parametrize('thes', [thesaurus_c(), thes_with_overlap(), thes_without_overlap()])
def test_from_shelf(thes, tmpdir):
    filename = str(tmpdir.join('test_shelf'))
    thes.to_shelf(filename)
    loaded_thes = Thesaurus.from_shelf_readonly(filename)
    for k, v in thes.iteritems():
        assert k in loaded_thes
        assert v == loaded_thes[k]


def test_allow_lexical_overlap(thes_with_overlap):
    assert len(thes_with_overlap) == 5
    assert len(thes_with_overlap['monetary/J_screw/N']) == 5
    assert len(thes_with_overlap['daily/J_pais/N']) == 5
    assert thes_with_overlap['japanese/J_yen/N'][0] == ('bundesbank/N_yen/N', 1.0)


# todo check this
def _assert_matrix_of_thesaurus_c_is_as_expected(matrix, rows, cols):
    # rows may come in any order
    assert set(rows) == set(['g/N', 'a/N', 'd/J', 'b/V', 'a/J_b/N'])
    # columns must be in alphabetical order
    assert cols == ['a/N', 'b/V', 'd/J', 'g/N', 'x/X']
    # test the vectors for each entry
    expected_matrix = np.array([
        [0.1, 0., 0.2, 0.8, 0.],  # ab
        [0., 0.1, 0.5, 0.3, 0.],  # a
        [0.1, 0., 0.3, 0.6, 0.],  # b
        [0.5, 0.3, 0., 0.7, 0.],  # d
        [0.3, 0.6, 0.7, 0., 0.9]  # g
    ])
    # put the rows in the matrix in the order in which they are in expected_matrix
    matrix_ordered_by_rows = matrix[np.argsort(np.array(rows)), :]
    assert_array_equal(matrix_ordered_by_rows, expected_matrix)


def test_to_sparse_matrix(thesaurus_c):
    matrix, cols, rows = thesaurus_c.to_sparse_matrix()
    matrix = matrix.A
    assert matrix.shape == (5, 5)

    _assert_matrix_of_thesaurus_c_is_as_expected(matrix, rows, cols)


def test_to_dissect_core_space(vectors_c):
    """
    :type vectors_c: Thesaurus
    """
    space = vectors_c.to_dissect_core_space()
    matrix = space.cooccurrence_matrix.mat.A
    _assert_matrix_of_thesaurus_c_is_as_expected(matrix, space.id2row, space.id2column)


def test_thesaurus_to_tsv(thesaurus_c, tmpdir):
    """

    :type thesaurus_c: Thesaurus
    :type tmpdir: py.path.local
    """
    # test columns(neighbours) are not reordered by Thesaurus
    filename = str(tmpdir.join('outfile.txt'))
    thesaurus_c.to_tsv(filename)
    t1 = Thesaurus.from_tsv([filename])
    assert t1._obj == thesaurus_c._obj


def test_vectors_to_tsv(vectors_c, tmpdir):
    """

    :type vectors_c: Vectors
    :type tmpdir: py.path.local
    """
    # these are feature vectors, columns(features) can be reordered
    filename = str(tmpdir.join('outfile.txt'))
    vectors_c.to_tsv(filename)
    t1 = Vectors.from_tsv([filename])

    # can't just assert t1 == thesaurus_c, because to_tsv may reorder the columns
    for k, v in vectors_c.iteritems():
        assert k in t1.keys()
        assert set(v) == set(vectors_c[k])


def test_to_dissect_sparse_files(vectors_c, tmpdir):
    """

    :type vectors_c: Thesaurus
    :type tmpdir: py.path.local
    """
    from composes.semantic_space.space import Space

    prefix = str(tmpdir.join('output'))
    vectors_c.to_dissect_sparse_files(prefix)
    # check that files are there
    for suffix in ['sm', 'rows', 'cols']:
        outfile = '{}.{}'.format(prefix, suffix)
        assert os.path.exists(outfile)
        assert os.path.isfile(outfile)

    # check that reading the files in results in the same matrix
    space = Space.build(data="{}.sm".format(prefix),
                        rows="{}.rows".format(prefix),
                        cols="{}.cols".format(prefix),
                        format="sm")

    matrix, rows, cols = space.cooccurrence_matrix.mat, space.id2row, space.id2column
    exp_matrix, exp_cols, exp_rows = vectors_c.to_sparse_matrix()

    assert exp_cols == cols
    assert exp_rows == rows
    assert_array_equal(exp_matrix.A, matrix.A)
    _assert_matrix_of_thesaurus_c_is_as_expected(matrix.A, rows, cols)
    _assert_matrix_of_thesaurus_c_is_as_expected(exp_matrix.A, exp_rows, exp_cols)


def test_load_with_column_filter():
    # test if constraining the vocabulary a bit correctly drops columns
    t = Thesaurus.from_tsv(tsv_files=['discoutils/tests/resources/exp0-0c.strings'],
                           column_filter=lambda x: x in {'a/N', 'b/V', 'd/J', 'g/N'})
    expected_matrix = np.array([
        [0.1, 0., 0.2, 0.8],  # ab
        [0., 0.1, 0.5, 0.3],  # a
        [0.1, 0., 0.3, 0.6],  # b
        [0.5, 0.3, 0., 0.7],  # d
        [0.3, 0.6, 0.7, 0.]  # g
    ])
    mat, cols, rows = t.to_sparse_matrix()
    assert set(cols) == {'a/N', 'b/V', 'd/J', 'g/N'}
    assert mat.shape == (5, 4)
    np.testing.assert_array_equal(expected_matrix.sum(axis=0)[np.newaxis], mat.sum(axis=0))

    # test if severely constraining the vocabulary a bit correctly drops columns AND rows
    t = Thesaurus.from_tsv(tsv_files=['discoutils/tests/resources/exp0-0c.strings'],
                           column_filter=lambda x: x in {'x/X'})
    mat, cols, rows = t.to_sparse_matrix()
    assert set(cols) == {'x/X'}
    assert mat.A == np.array([0.9])


def test_load_with_row_filter():
    # test if constraining the vocabulary a bit correctly drops columns
    t = Thesaurus.from_tsv(tsv_files=['discoutils/tests/resources/exp0-0c.strings'],
                           row_filter=lambda x, y: x in {'a/N', 'd/J', 'g/N'})
    expected_matrix = np.array([
        [0., 0.1, 0.5, 0.3, 0.],  # a
        [0.5, 0.3, 0., 0.7, 0.],  # d
        [0.3, 0.6, 0.7, 0., 0.9]  # g
    ])
    mat, cols, rows = t.to_sparse_matrix()
    assert set(cols) == {'a/N', 'b/V', 'd/J', 'g/N', 'x/X'}
    assert set(rows) == {'a/N', 'd/J', 'g/N'}
    np.testing.assert_array_equal(expected_matrix.sum(axis=0)[np.newaxis], mat.sum(axis=0))


def test_load_with_max_num_neighbours():
    t = Thesaurus.from_tsv(tsv_files=['discoutils/tests/resources/exp0-0c.strings'],
                           max_neighbours=1)
    assert all(len(neigh) == 1 for neigh in t.values())
    mat, cols, rows = t.to_sparse_matrix()
    assert set(rows) == set(['g/N', 'a/N', 'd/J', 'b/V', 'a/J_b/N'])
    assert cols == ['d/J', 'g/N', 'x/X']


def test_max_num_neighbours_and_no_lexical_overlap():
    # max_neighbours filtering should kick in after lexical overlap filtering
    t = Thesaurus.from_tsv(tsv_files=['discoutils/tests/resources/exp0-0d.strings'],
                           allow_lexical_overlap=False)
    assert len(t) == 4
    assert len(t['trade/N_law/N']) == 1
    assert len(t['prince/N_aziz/N']) == 3
    assert len(t['important/J_country/N']) == 5
    assert len(t['foreign/J_line/N']) == 3

    t = Thesaurus.from_tsv(tsv_files=['discoutils/tests/resources/exp0-0d.strings'],
                           allow_lexical_overlap=False,
                           max_neighbours=1)
    assert len(t) == 4
    assert len(t['trade/N_law/N']) == 1
    assert t['trade/N_law/N'][0][0] == 'product/N_line/N'
    assert len(t['prince/N_aziz/N']) == 1
    assert len(t['important/J_country/N']) == 1
    assert len(t['foreign/J_line/N']) == 1

    t = Thesaurus.from_tsv(tsv_files=['discoutils/tests/resources/exp0-0d.strings'])
    assert t['trade/N_law/N'][0][0] == 'law/N'
    assert t['trade/N_law/N'][4][0] == 'product/N_line/N'

    t = Thesaurus.from_tsv(tsv_files=['discoutils/tests/resources/exp0-0d.strings'],
                           allow_lexical_overlap=True,
                           max_neighbours=1)
    assert t['trade/N_law/N'][0][0] == 'law/N'
    assert len(t['trade/N_law/N']) == 1


class TestLoad_thesauri(TestCase):
    def setUp(self):
        """
        Sets the default parameters of the tokenizer and reads a sample file
        for processing
        """

        self.params = {
            'tsv_files': ['discoutils/tests/resources/exp0-0a.strings'],
            'sim_threshold': 0,
            # 'k': 10,
            'include_self': False
        }
        self.thesaurus = Thesaurus.from_tsv(**self.params)

    def _reload_thesaurus(self):
        self.thesaurus = Thesaurus.from_tsv(**self.params)

    def test_empty_thesaurus(self):
        self.params['tsv_files'] = []
        self._reload_thesaurus()
        self._reload_and_assert(0, 0)

        # should raise KeyError for unknown tokens
        with self.assertRaises(KeyError):
            self.thesaurus['kasdjhfka']

    def _reload_and_assert(self, entry_count, neighbour_count):
        th = Thesaurus.from_tsv(**self.params)
        all_neigh = [x for v in th.values() for x in v]
        self.assertEqual(len(th), entry_count)
        self.assertEqual(len(all_neigh), neighbour_count)
        return th

    def test_from_dict(self):
        from_dict = Thesaurus(self.thesaurus._obj)
        self.assertDictEqual(self.thesaurus._obj, from_dict._obj)

    def test_from_shelved_dict(self):
        filename = 'thesaurus_unit_tests.tmp'
        self.thesaurus.to_shelf(filename)

        d = shelve.open(filename, flag='r')  # read only
        from_shelf = Thesaurus(d)
        for k, v in self.thesaurus.iteritems():
            self.assertEqual(self.thesaurus[k], from_shelf[k])

        # check mutability
        self.thesaurus['some_value'] = ('should be possible', 0)

        def modify():
            from_shelf['some_value'] = ('should not be possible', 0)

        self.assertRaises(DBAccessError, modify)

        # tear down
        self.assertTrue(os.path.exists(filename))
        d.close()
        if os.path.exists(filename):
            os.unlink(filename)

    def test_sim_threshold(self):
        for i, j, k in zip([0, .39, .5, 1], [7, 3, 3, 0], [14, 4, 3, 0]):
            self.params['sim_threshold'] = i
            self._reload_thesaurus()
            self._reload_and_assert(j, k)


    def test_include_self(self):
        for i, j, k in zip([False, True], [7, 7], [14, 21]):
            self.params['include_self'] = i
            self._reload_thesaurus()
            th = self._reload_and_assert(j, k)

            for entry, neighbours in th.items():
                self.assertIsInstance(entry, str)
                self.assertIsInstance(neighbours, list)
                self.assertIsInstance(neighbours[0], tuple)
                if i:
                    self.assertEqual(entry, neighbours[0][0])
                    self.assertEqual(1, neighbours[0][1])
                else:
                    self.assertNotEqual(entry, neighbours[0][0])
                    self.assertGreaterEqual(1, neighbours[0][1])


    def test_iterate_nonoverlapping_pairs(self):
        inp = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        output1 = [x for x in walk_nonoverlapping_pairs(inp, 1)]
        self.assertListEqual([(1, 2), (3, 4), (5, 6), (7, 8)], output1)

        output1 = [x for x in walk_nonoverlapping_pairs(inp, 1, max_pairs=2)]
        self.assertListEqual([(1, 2), (3, 4)], output1)

        output1 = [x for x in walk_nonoverlapping_pairs(inp, 1, max_pairs=-2)]
        self.assertListEqual([], output1)