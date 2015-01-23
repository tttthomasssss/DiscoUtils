import pytest
import numpy as np
import scipy.sparse as sp

from discoutils.reduce_dimensionality import _do_svd_single, filter_out_infrequent_entries, do_svd
from discoutils.thesaurus_loader import Vectors
from discoutils.tests.test_thesaurus import thesaurus_c # this is used, do not remove


DIM = 100
__author__ = 'mmb28'


@pytest.fixture(scope='module')
def dense_matrix():
    a = np.random.random((DIM, DIM))
    a[a < 0.4] = 0
    return a


@pytest.fixture(scope='module')
def sparse_matrix(dense_matrix):
    matrix = sp.csr_matrix(dense_matrix)
    assert matrix.nnz < DIM ** 2
    return matrix


def test_do_svd_single_dense(dense_matrix):
    for i in range(10, 51, 10):
        reducer, matrix = _do_svd_single(dense_matrix, i)
        matrix1 = reducer.inverse_transform(matrix)
        assert matrix.shape == (DIM, i)
        assert matrix1.shape == dense_matrix.shape


def test_do_svd_single_sparse(sparse_matrix):
    test_do_svd_single_dense(sparse_matrix)


@pytest.mark.parametrize(
    ('first', 'second', 'exp_row_len'),
    (
        ('c', 'c', 5), # easy case, vocabulary matches
        ('b', 'b', 4), # easy case, vocabulary matches
        ('b', 'c', 7), # unseen features introduced. The union of the 2 thesauri has 7 valid entries
        ('c', 'b', 7),  # some seen features missing
    ),
)
def test_application_after_learning(tmpdir, first, second, exp_row_len):
    """
    Test of applying a learnt SVD to another matrix works. We are mostly interested if
    matrix dimensions match- no exception should be raised. Other than that,
    this is a useless test
    """
    tmpfile = tmpdir.join('tmp.thesaurus')
    do_svd('discoutils/tests/resources/exp0-0%s.strings' % first,
           tmpfile,
           reduce_to=[2], # some small number, not what we are testing for here
           apply_to='discoutils/tests/resources/exp0-0%s.strings' % second)

    # when made into a thesaurus, the reduced matrix will have some duplicates
    # these will be summed out, leaving us with a matrix of a specific size
    t = Vectors.from_tsv(str(tmpfile) + '-SVD2.events.filtered.strings',
                           lowercasing=False, gzipped=True)
    mat, cols, rows = t.to_sparse_matrix()
    assert mat.shape == (exp_row_len, 2)


def test_application_after_learning_with_selective_write(tmpdir):
    """
    Test if when SVD is trained on matrix A and applied to matrix B, and
    it is requested that just the reduced version of only A or B is output,
    the shape of the output is right
    """
    tmpfile = tmpdir.join('tmp.thesaurus')
    for w, exp_row_len in zip([1, 2, 3], [4, 5, 7]):
        do_svd('discoutils/tests/resources/exp0-0b.strings',
               tmpfile,
               reduce_to=[2], # some small number, not what we are testing for here
               apply_to='discoutils/tests/resources/exp0-0c.strings',
               write=w)
        t = Vectors.from_tsv(str(tmpfile) + '-SVD2.events.filtered.strings',
                             lowercasing=False, gzipped=True)
        mat, _, _ = t.to_sparse_matrix()
        assert mat.shape == (exp_row_len, 2)


@pytest.fixture(scope='module')
def all_cols(thesaurus_c):
    _, cols, _ = thesaurus_c.to_sparse_matrix()
    assert len(cols) == 5
    return cols


@pytest.mark.parametrize(
    ('feature_type_limits', 'expected_shape', 'missing_columns'),
    (
        ([('N', 2), ('V', 2), ('J', 2), ('AN', 2)], (5, 5), []), # nothing removed
        ([('N', 1), ('V', 2), ('J', 2), ('AN', 2)], (4, 5), []), # just row a/N should drop out
        ([('N', 0), ('V', 2), ('J', 2), ('AN', 2)], (3, 4), ['x/X']), # rows a and g, column x should drop out
        ([('V', 1)], (1, 3), ['b/V', 'x/X']),  # just the one verb should remain, with its three features
    ),
)
def test_filter_out_infrequent_entries(thesaurus_c, all_cols, feature_type_limits, expected_shape, missing_columns):
    mat, pos_tags, rows, cols = filter_out_infrequent_entries(feature_type_limits, thesaurus_c)
    assert mat.shape == expected_shape
    assert set(all_cols) - set(missing_columns) == set(cols)


def _read_and_strip_lines(input_file):
    with open(input_file) as infile:
        lines = infile.readlines()
    lines = map(str.strip, lines)
    lines = [x for x in lines if x]
    return lines
