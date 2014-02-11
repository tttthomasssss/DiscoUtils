import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

import logging
import scipy, time
import numpy as np
from operator import itemgetter
from sklearn.decomposition import TruncatedSVD
from discoutils.tokens import DocumentFeature
from discoutils.thesaurus_loader import Thesaurus
from discoutils.io_utils import write_vectors_to_disk

try:
    import cPickle as pickle
except ImportError:
    import pickle


def _filter_out_infrequent_entries(desired_counts_per_feature_type, thesaurus):
    logging.info('Converting thesaurus to sparse matrix')
    mat, cols, rows = thesaurus.to_sparse_matrix()
    logging.info('Loaded a data matrix of shape %r', mat.shape)
    # convert to document feature for access to PoS tag
    document_features = [DocumentFeature.from_string(r) for r in rows]
    # don't want to do dimensionality reduction on composed vectors
    feature_types = [sorted_idx_and_pos_matching.type for sorted_idx_and_pos_matching in document_features]
    assert all(x == '1-GRAM' or x == 'AN' or x == 'NN' for x in feature_types)
    # get the PoS tags of each row in the matrix
    pos_tags = np.array([df.tokens[0].pos if df.type == '1-GRAM' else df.type for df in document_features])
    # find the rows of the matrix that correspond to the most frequent nouns, verbs, ...,
    # as measured by sum of feature counts. This is Byblo's definition of frequency (which is in fact a marginal),
    # but it is strongly correlated with one normally thinks of as entry frequency
    desired_rows = []
    for desired_pos, desired_count in desired_counts_per_feature_type:
        row_of_current_pos = pos_tags == desired_pos  # what rows are the right PoS tags at, boolean mask array
        # indices of the array sorted by row sum, and where the pos == desired_pos
        if desired_count > 0:
            sorted_idx_by_sum = np.ravel(mat.sum(1)).argsort()
            row_of_current_pos = row_of_current_pos[sorted_idx_by_sum]
            sorted_idx_and_pos_matching = sorted_idx_by_sum[row_of_current_pos]
            # slice off the top desired_count and store them
            desired_rows.extend(list(sorted_idx_and_pos_matching[-desired_count:]))
        else:
            #do not include
            pass

        logging.info('Frequency filter keeping %d/%d %s entries ', desired_count,
                     sum(row_of_current_pos), desired_pos)
    desired_rows = sorted(desired_rows)
    # check that the pos tag of each selected entry is what we think it is
    # for k, v in pos_to_rows.iteritems():
    #     assert all(k == x for x in pos_tags[v])

    # remove the vectors for infrequent entries, update list of pos tags too
    mat = mat[desired_rows, :]
    rows = itemgetter(*desired_rows)(document_features)
    pos_tags = pos_tags[desired_rows]
    # removing rows may empty some columns, remove these as well. This is probably not very like to occur as we have
    # already filtered out infrequent features, so the column count will stay roughly the same
    desired_cols = np.ravel(mat.sum(0)) > 0
    mat = mat[:, desired_cols]
    cols = np.array(cols)[desired_cols]
    logging.info('Selected only the most frequent entries, matrix size is now %r', mat.shape)
    return mat, pos_tags, rows, cols


def _do_svd_single(mat, n_components):
    if n_components > mat.shape[1]:
        logging.error('Cannot reduce dimensionality from %d to %d', mat.shape[1], n_components)
        return None, None

    method = TruncatedSVD(n_components)
    logging.info('Reducing dimensionality of matrix of shape %r', mat.shape)
    start = time.time()
    reduced_mat = method.fit_transform(mat)
    end = time.time()
    logging.info('Reduced using {} from shape {} to shape {} in {} seconds'.format(method,
                                                                                   mat.shape,
                                                                                   reduced_mat.shape,
                                                                                   end - start))
    return method, reduced_mat


def _write_to_disk(reduced_mat, method, prefix, rows):
    features_file = prefix + '.features.filtered.strings'
    events_file = prefix + '.events.filtered.strings'
    entries_file = prefix + '.entries.filtered.strings'
    model_file = prefix + '.model.pkl'
    write_vectors_to_disk(reduced_mat, rows,
                          ['SVD:feat{0:05d}'.format(i) for i in range(reduced_mat.shape[1])],
                          events_file, features_file, entries_file)

    # disabled because it causes a crash with large objects
    # see http://bugs.python.org/issue11564
    #with open(model_file, 'w') as outfile:
    #    pickle.dump(method, outfile)


def do_svd(input_paths,
           output_prefix,
           desired_counts_per_feature_type=[('N', 8), ('V', 4), ('J', 4), ('RB', 2), ('AN', 2)],
           reduce_to=[3, 10, 15],
           apply_to=[]):
    """

    Performs truncated SVD. A copy of the trained sklearn SVD estimator will be also be saved

    :param input_paths: list of files containing vectors in TSV format. All vectors will be reduced together.
    :type input_paths: list
    :param output_prefix: Where to output the reduced files. An extension will be added.
    :param desired_counts_per_feature_type: how many entries to keep of each DocumentFeature type, by frequency. This
     is the PoS tag for unigram features and the feature type otherwise. For instance, pass in [('N', 2), ('AN', 0)] to
    select 2 unigrams of PoS N and 0 bigrams of type adjective-noun. Types that are not explicitly given a positive
    desired count are treated as if the desired count is 0.
    :param reduce_to: list of integers, what dimensionalities to reduce to
    :param apply_to: list of file paths. After SVD has been trained on input_paths, it can be applied to
    apply_to. Output will be writen to the same file

    :raise ValueError: If the loaded thesaurus is empty
    """

    thesaurus = Thesaurus.from_tsv(input_paths, aggressive_lowercasing=False)
    if not thesaurus:
        raise ValueError('Empty thesaurus %r', input_paths)
    mat, pos_tags, rows, cols = _filter_out_infrequent_entries(desired_counts_per_feature_type, thesaurus)
    if apply_to:
        thes_to_apply_to = Thesaurus.from_tsv(apply_to, aggressive_lowercasing=False, vocabulary=set(cols))
        # get the names of each thesaurus entry
        extra_rows = [x for x in thes_to_apply_to.keys()]
        # vectorize second matrix with the vocabulary (columns) of the first thesaurus to ensure shapes match
        # "project" second thesaurus into space of first thesaurus
        extra_matrix = thesaurus.v.transform([dict(fv) for fv in thes_to_apply_to.itervalues()])
        # make sure the shape is right
        assert extra_matrix.shape[1] == mat.shape[1]

    for n_components in reduce_to:
        method, reduced_mat = _do_svd_single(mat, n_components)
        if not method:
            continue
        if apply_to:
            logging.info('Applying learned SVD transform to matrix of shape %r', extra_matrix.shape)
            # apply learned transform to new data and append to old data
            reduced_mat = np.vstack((reduced_mat, method.transform(extra_matrix)))
            # also extend the list of names
            rows = list(rows) + [DocumentFeature.from_string(x) for x in extra_rows]

        path = '{}-SVD{}'.format(output_prefix, n_components)
        _write_to_disk(scipy.sparse.coo_matrix(reduced_mat), method, path, rows)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d)\t%(levelname)s : %(message)s")

    # in_paths = dump.giga_paths
    # out_prefixes = [path.split('.')[0] for path in in_paths]
    # do_svd(['../FeatureExtractionToolkit/exp10-12/exp10.events.filtered.strings'], 'wtf',
    # desired_counts_per_feature_type=[('N', 8000), ('V', 4000), ('J', 4000), ('RB', 200), ('AN', 20000),
    #                                  ('NN', 20000)],
    # reduce_to=[30, 300, 1000],
    # apply_to=['../FeatureExtractionToolkit/exp10-12/exp10.events.filtered.strings'])