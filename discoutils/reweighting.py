__author__ = 'mmb28'

import numpy as np
import scipy.sparse as sp
import logging


def ppmi_sparse_matrix(mat:sp.csr_matrix):
    """
    Source: https://github.com/Bollegala/svdmi/blob/master/src/svdmi.py
    Compute the PPMI values for the raw co-occurrence matrix.
    PPMI values will be written to mat and it will get overwritten.
    """
    logging.info('Doing PPMI on matrix of size %r', mat.shape)
    (nrows, ncols) = mat.shape
    col_totals = np.zeros(ncols, dtype=np.float)
    for j in range(0, ncols):
        col_totals[j] = np.sum(mat[:, j].data)
    N = np.sum(col_totals)
    for i in range(0, nrows):
        row = mat[i, :]
        rowTotal = np.sum(row.data)
        for j in row.indices:
            val = np.log((mat[i, j] * N) / (rowTotal * col_totals[j]))
            mat[i, j] = max(0, val)
    res = sp.csr_matrix(mat)  # make a new matrix to remove any zeroes that were just created
    if res.shape != mat.shape:
        # this may happen if a feature occurs with all entries, so its PPMI weight will be 0 and
        # it will effectively be removed from the list of features (vocabulary). This is problematics because the
        # vocabulary is not updated here.
        logging.error('Shape after PPMI is %r', res.shape)
        raise ValueError('PPMI changed shape of matrix!')
    return res