import gzip
from itertools import groupby, chain
import logging
from operator import itemgetter
import os
from scipy.sparse import isspmatrix_coo, issparse
import numpy as np
import six

__author__ = 'mmb28'


def write_vectors_to_disk(matrix, row_index, column_index, vectors_path, features_path='', entries_path='',
                          entry_filter=lambda x: True, gzipped=False):
    """
    Converts a matrix and its associated row/column indices to a Byblo compatible entries/features/event files,
    possibly applying a tranformation function to each entry.

    :param matrix: data matrix of size (n_entries, n_features) in scipy.sparse.coo format
    :type matrix: scipy.sparse.coo_matrix
    :param row_index: a collection of DocumentFeature-s representing entry names. `row_index[N]` should return the
     feature whose vector is stored in row N of `matrix`
    :type row_index: thesisgenerator.plugins.tokenizer.DocumentFeature
    :param column_index: sorted list of feature names
    :param features_path: str, where to write the Byblo features file. If the entry_filter removes all entries
    this file will not be written, i.e. the file will not be created at all if there's nothing to put in it
    :param entries_path: str, where to write the Byblo entries file. If the entry_filter removes all entries
    this file will not be written.
    :param vectors_path: where to write the Byblo events file
    :type vectors_path: string of file-like. If it evaluates to True progress messages will be printed
    :param entry_filter: callable, called for each entry. Takes a single DocumentFeature parameter. Returns true
    if the entry has to be written and false if the entry has to be ignored. Defaults to True.
    """
    import numpy as np

    if not any([vectors_path, features_path, entries_path]):
        raise ValueError('At least one of vectors_path, features_path or entries_path required')

    if not isspmatrix_coo(matrix):
        logging.error('Expected a scipy.sparse.coo matrix, got %s', type(matrix))
        raise ValueError('Wrong matrix type')
    if (len(row_index), len(column_index)) != matrix.shape:
        logging.error('Matrix shape is wrong, expected %dx%s, got %r', len(row_index), len(column_index), matrix.shape)
        raise ValueError('Matrix shape does not match row_index/column_index size')

    accepted_entry_counts = {}
    matrix_data = zip(matrix.row, matrix.col, matrix.data)
    accepted_rows = []

    logging.info('Writing events to %s', vectors_path)
    if isinstance(vectors_path, six.string_types):
        if gzipped:
            outfile = gzip.open(vectors_path, 'w')
        else:
            outfile = open(vectors_path, 'w')
    elif hasattr(vectors_path, 'write'):
        outfile = vectors_path
    else:
        raise ValueError('vectors_path: expected str or file-like, got %s' % type(vectors_path))

    for row_num, column_ids_and_values in groupby(matrix_data, itemgetter(0)):
        entry = row_index[row_num]
        if entry_filter(entry):
            if entry not in accepted_entry_counts:  # guard against duplicated vectors
                accepted_rows.append(row_num)
                features_and_counts = [(column_index[feat], count) for _, feat, count in column_ids_and_values \
                                       if not -0.0001 < count < 0.0001]  # remove almost zero feature counts
                if not features_and_counts:
                    continue
                s = '%s\t%s\n' % (entry, '\t'.join(map(str, chain.from_iterable(features_and_counts))))
                outfile.write(s.encode('utf8') if gzipped else s)
                accepted_entry_counts[entry] = sum(x[1] for x in features_and_counts)
            if row_num % 20000 == 0 and outfile:
                logging.info('Processed %d vectors', row_num)

    outfile.close()

    if entries_path and accepted_entry_counts:
        logging.info('Writing entries to %s', entries_path)
        with open(entries_path, 'w') as outfile:
            for entry, count in accepted_entry_counts.items():
                outfile.write('%s\t%f\n' % (entry, count))

    if features_path and accepted_rows:  # guard against empty files
        logging.info('Writing features to %s', features_path)
        with open(features_path, 'w') as outfile:
            feature_sums = np.array(matrix.tocsr()[accepted_rows].sum(axis=0))[0, :]
            for feature, count in zip(column_index, feature_sums):
                if -1e-5 < count < 1e-5:
                    logging.warning('Feature %s does not occur in vector set', feature)
                else:
                    outfile.write('%s\t%f\n' % (feature, count))


def write_vectors_to_hdf(matrix, row_index, column_index, events_path):
    import pandas as pd

    logging.info('Writing vectors of shape %r to %s', matrix.shape, events_path)
    if isinstance(row_index, dict):
        # row_index is a dict, let's make it into a list
        ri = list(range(len(row_index)))  # mega inefficient, but numpy str arrays confuse me
        for phrase, idx in row_index.items():
            try:
                str(phrase).encode('ascii')
                ri[idx] = str(phrase)
            except UnicodeEncodeError as e:
                # pandas doesnt like non-unicode keys in index; mark such phrases for removal
                ri[idx] = 'THIS_IS_FUCKED_YO_%d' % idx
    else:
        ri = list(map(str, row_index))
    old_shape = matrix.shape
    # remove phrases that arent ascii-only
    to_keep = np.array([False if str(x).startswith('THIS_IS_FUCKED_YO_') else True for x in ri])
    matrix = matrix.A if issparse(matrix) else matrix
    matrix = matrix[to_keep, :]
    ri = np.array(ri)[to_keep]
    if old_shape != matrix.shape:
        logging.info('Removing non-ascii phrases. Matrix shape was %r, is now %r', old_shape, matrix.shape)

    df = pd.DataFrame(matrix, index=ri, columns=map(str, column_index))
    df[df.columns[:4]].to_html('tmp.html')
    df[df.columns[:4]].head(7550).tail(50).to_html('tmp_small.html')
    if os.path.exists(events_path):
        # PyTables fails if the file exist, but is not and HDF store. Remove the file
        os.unlink(events_path)
    df.to_hdf(events_path, 'matrix', complevel=9, complib='zlib')


def reformat_entries(filename, suffix, function, separator='\t'):
    # todo unit test
    """
    Applies a function to the first column of a file
    :param filename: File to apply transformation to.
    :param suffix: suffix to append to the output file
    :param function: Function to apply, takes and returns a single string.
    :param separator: The columns in the file are separated by this.
    """

    # shutil.copy(filename, filename + '.bak')
    outname = '{}{}'.format(filename, suffix)
    with open(filename) as infile, open(outname, 'w') as outfile:
        for line in infile:
            fields = line.split(separator)
            if len(fields) < 2:
                # some line may contain an entry but no features
                continue
            fields[0] = function(fields[0])
            outfile.write(separator.join(fields))
    return outname


def clean(entry):
    """
    CONVERT A FILE FROM JULIE'S FORMAT TO MINE
    absurdity/N:amod-DEP:total/J -> total/J_absurdity/N
    academy/N:nn-HEAD:award/N -> academy/N_award/N
    :param entry:
    :return:
    """
    import re

    pattern = re.compile('(\S+):(\S+):(\S+)')
    a, relation, b = pattern.match(entry).groups()
    if relation == 'amod-HEAD':
        return '{}_{}'.format(a, b)
    elif relation == 'amod-DEP':
        return '{}_{}'.format(b, a)
    elif relation == 'nn-HEAD':
        return '{}_{}'.format(a, b)
    elif relation == 'nn-DEP':
        return '{}_{}'.format(b, a)
    else:
        raise ValueError('Can not convert entry %s' % entry)
