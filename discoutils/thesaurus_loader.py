# coding=utf-8
from collections import Counter
import contextlib
import gzip
import logging
import os
import shelve
import numpy as np
import six
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from scipy.sparse import csr_matrix, issparse, coo_matrix
from discoutils.tokens import DocumentFeature
from discoutils.collections_utils import walk_nonoverlapping_pairs
from discoutils.io_utils import write_vectors_to_disk, write_vectors_to_hdf
from discoutils.misc import is_gzipped, is_hdf
from sklearn.neighbors import NearestNeighbors

from functools import lru_cache


class Thesaurus(object):
    def __init__(self, d, immutable=True):

        """
         A container that can read Byblo-formatted  sims files. Each entry can be of the form

            'water/N': [('horse/N', 0.5), ('earth/N', 0.4)]

        i.e. entry: [(neighbour, similarity), ...]

        Use `Vectors` to store feature vectors

        :param d: a dictionary that serves as a basis. All magic method call on `Thesaurus` objects are forwarded
        to this dict so users can think of this class as a normal dict.
        :param immutable: if true, Thesaurus['a'] = 1 will raise a ValueError
        """
        self._obj = d  # do not rename
        self.immutable = immutable

    @classmethod
    def from_shelf_readonly(cls, shelf_file_path):
        return Thesaurus(shelve.open(shelf_file_path, flag='r'))  # read only

    @classmethod
    def remove_overlapping_neighbours(cls, entry, to_insert):
        """

        :type entry: DocumentFeature or str
        :type to_insert: list of (str, float) tuples
        """
        if isinstance(entry, (six.string_types, six.text_type)):
            entry = DocumentFeature.from_string(entry)
        features = [(DocumentFeature.from_string(x[0]), x[1]) for x in to_insert]
        to_insert = [(str(f[0]), f[1]) for f in features
                     if not any(t in entry.tokens for t in f[0].tokens)]
        return to_insert

    @classmethod
    def from_tsv(cls, tsv_file, sim_threshold=0, include_self=False,
                 lowercasing=False, ngram_separator='_', pos_separator='/', allow_lexical_overlap=True,
                 row_filter=lambda x, y: True, column_filter=lambda x: True, max_len=50,
                 max_neighbours=1e8, merge_duplicates=False, immutable=True,
                 enforce_word_entry_pos_format=True, **kwargs):
        """
        Create a Thesaurus by parsing a Byblo-compatible TSV files (events or sims).
        If duplicate values are encoutered during parsing, only the latest will be kept.

        :param tsv_file: path to input TSV file. May be gzipped.
        :type tsv_file:  str
        :param sim_threshold: min similarity between an entry and its neighbour for the neighbour to be included
        :type sim_threshold: float
        :param include_self: whether to include self as nearest neighbour.
        :type include_self: bool
        :param lowercasing: if true, most of what is read will be lowercased (excluding PoS tags), so
            Cat/N -> cat/N. This is desirable when reading thesauri with this class. If False, no lowercasing
            will take place. This might be desirable when readings feature lists or already lowercased neighbour
            lists. FET + Byblo thesauri are already lowercased.
        :type lowercasing: bool
        :param ngram_separator: When n_gram entries are read in, what are the indidivual tokens separated by
        :param column_filter: A function that takes a string (column in the file) and returns whether or not
        the string should be kept
        :param row_filter: takes a string and its corresponding DocumentFeature and determines if it should be loaded.
        If `enforce_word_entry_pos_format` is `False`, the second parameter to this function will be `None`
        :param allow_lexical_overlap: whether neighbours/features are allowed to overlap lexically with the entry
        they are neighbours/features of. OTE: THE BEHAVIOUR OF THIS PARAMETER IS SLIGHTLY DIFFERENT FROM THE EQUIVALENT
        IN VECTORS. SEE COMMENT THERE.
        :param max_len: maximum length (in characters) of permissible **entries**. Longer entries are ignored.
        :param max_neighbours: maximum neighbours per entry. This is applied AFTER the filtering defined by
        column_filter and allow_lexical_overlap is finished.
        :param merge_duplicates: whether to raise en error if multiple entries exist, or concatenate/add them together.
        The former is appropriate for `Thesaurus`, and the latter for `Vectors`
        :param enforce_word_entry_pos_format: if true, entries that are not in a `word/POS` format are skipped. This
        must be true for `allow_lexical_overlap` to work.
        """

        if not tsv_file:
            raise ValueError("No thesaurus specified")

        DocumentFeature.recompile_pattern(pos_separator=pos_separator, ngram_separator=ngram_separator)
        to_return = dict()
        logging.info('Loading thesaurus %s from disk', tsv_file)

        if not allow_lexical_overlap:
            logging.warning('DISALLOWING LEXICAL OVERLAP')

        if not allow_lexical_overlap and not enforce_word_entry_pos_format:
            raise ValueError('allow_lexical_overlap requires entries to be converted to a DocumentFeature. '
                             'Please enable enforce_word_entry_pos_format')
        FILTERED = '___FILTERED___'.lower()

        gzipped = is_gzipped(tsv_file)
        if gzipped:
            logging.info('Attempting to read a gzipped file')
            fhandle = gzip.open(tsv_file)
        else:
            fhandle = open(tsv_file)

        with fhandle as infile:
            for line in infile.readlines():
                if gzipped:
                    # this is a byte steam, needs to be decoded
                    tokens = line.decode('UTF8').strip().split('\t')
                else:
                    tokens = line.strip().split('\t')

                if len(tokens) % 2 == 0:
                    # must have an odd number of things, one for the entry
                    # and pairs for (neighbour, similarity)
                    logging.warning('Skipping dodgy line in thesaurus file: %s\n %s', tsv_file, line)
                    continue

                if tokens[0] != FILTERED:
                    key = DocumentFeature.smart_lower(tokens[0], lowercasing)
                    dfkey = DocumentFeature.from_string(key) if enforce_word_entry_pos_format else None

                    if enforce_word_entry_pos_format and dfkey.type == 'EMPTY':
                        # do not load things in the wrong format, they'll get in the way later
                        # logging.warning('%s is not in the word/POS format, skipping', tokens[0])
                        continue

                    if (not row_filter(key, dfkey)) or len(key) > max_len:
                        logging.debug('Skipping entry for %s', key)
                        continue

                    to_insert = [(DocumentFeature.smart_lower(word, lowercasing), float(sim))
                                 for (word, sim) in walk_nonoverlapping_pairs(tokens, 1)
                                 if word.lower() != FILTERED and column_filter(word) and float(sim) > sim_threshold]

                    if not allow_lexical_overlap:
                        to_insert = cls.remove_overlapping_neighbours(dfkey, to_insert)

                    if len(to_insert) > max_neighbours:
                        to_insert = to_insert[:max_neighbours]

                    if include_self:
                        to_insert.insert(0, (key, 1.0))

                    # the steps above may filter out all neighbours of an entry. if this happens,
                    # do not bother adding it
                    if len(to_insert) > 0:
                        if key in to_return:  # this is a duplicate entry, merge it or raise an error
                            if merge_duplicates:
                                logging.debug('Multiple entries for "%s" found. Merging.', tokens[0])
                                c = Counter(dict(to_return[key]))
                                c.update(dict(to_insert))
                                to_return[key] = [(k, v) for k, v in c.items()]
                            else:
                                raise ValueError('Multiple entries for "%s" found.' % tokens[0])
                        else:
                            to_return[key] = to_insert
                    else:
                        logging.warning('Nothing survived filtering for %r', key)
        return Thesaurus(to_return, immutable=immutable)

    def to_shelf(self, filename):
        """
        Uses the shelf module to persist this object to a file
        """
        logging.info('Shelving thesaurus of size %d to %s', len(self), filename)
        d = shelve.open(filename, flag='c')  # read and write
        for entry, features in self.items():
            d[str(entry)] = features
        d.close()

    def to_tsv(self, filename, gzipped=False):
        """
        Writes this thesaurus to a Byblo-compatible sims file like the one it was most likely read from.  Neighbours
        are written in the order that they appear in.
        :param filename: file to write to
        :return: the file name
        """
        logging.warning('row_transform and entry_filter options are ignored in order to use preserve_order')
        if gzipped:
            f = gzip.open(filename, 'w')
        else:
            f = open(filename, 'w')
        with contextlib.closing(f) as outfile:
            for entry, vector in self._obj.items():
                features_str = '\t'.join(['%s\t%f' % foo for foo in vector])
                outfile.write('%s\t%s\n' % (entry, features_str))
        return filename

    def to_sparse_matrix(self, row_transform=None, dtype=np.float):
        """
        Converts the vectors held in this object to a scipy sparse matrix. Raises a ValueError if
        the thesaurus is empty

        WARNING: This method doesn't make much sense for a Thesaurus and belongs in Vectors. I'm leaving it
        here as some existing Thesaurus tests rely on it.

        :return: a tuple containing
            1) the sparse matrix, in which rows correspond to the order of this object's items()
            2) a **sorted** list of all features (column labels of the matrix).
            3) a list of all entries (row labels of the matrix)
        :rtype: tuple
        """
        from sklearn.feature_extraction import DictVectorizer

        self.v = DictVectorizer(sparse=True, dtype=dtype)

        # order in which rows are iterated is not guaranteed if the dict if modified, but we're not doing that,
        # so it's all fine
        mat = self.v.fit_transform([dict(fv) for fv in self.values()])
        rows = [k for k in self.keys()]
        if row_transform:
            rows = map(row_transform, rows)

        return mat, self.v.feature_names_, rows

    def keys(self):
        return self._obj.keys()

    def values(self):
        return self._obj.values()

    def items(self):
        return self._obj.items()

    def __len__(self):
        return len(self._obj)

    def __setstate__(self, d):
        """
        Defining this explicitly is required for pickling to work. If the constructor does work other than saving
        the provided parameters to field pickling will not work, because __init__ is not called upon unpickling.
        The fiels computed by the constructor will therefore not exist and the state of the class will not be restored
        fully. However, the state of the class is contained fully in the parameter of this method, we just need to
        save it, as the constructor would have done had it been invoked.

        In this class the issue manifests in a weird way. __getattr__ may be called during unpickling, before the
        class invariant is established (as the constructor isn't called). This class' __getattr__ delegates to
        self._obj, which does not exist, so __getattr__ is called again!

        See https://docs.python.org/2/library/pickle.html#pickle-inst
        """
        self.__dict__.update(d)

    def __setitem__(self, key, value):
        if self.immutable:
            raise ValueError('This object is immutable')
        if isinstance(key, DocumentFeature):
            key = str(key)
        self._obj[key] = value

    def __delitem__(self, key):
        """
        Deletes key from the list of entries in the thesaurus and the matrix
        :param key:
        :type key:
        :return:
        :rtype:
        """
        if isinstance(key, DocumentFeature):
            item = str(key)

        del self._obj[key]
        if hasattr(self, 'matrix'):
            mask = np.ones(self.matrix.shape[0], dtype=bool)
            mask[self.name2row[key]] = False
            self.matrix = self.matrix[mask, :]

    def __getitem__(self, item):
        if isinstance(item, DocumentFeature):
            item = str(item)
        return self._obj[item]

    get_nearest_neighbours = __getitem__

    def __contains__(self, item):
        if isinstance(item, DocumentFeature):
            item = str(item)
        return item in self._obj

    def __len__(self):
        return len(self._obj)


class Vectors(Thesaurus):
    def __init__(self, d, immutable=True, allow_lexical_overlap=True,
                 matrix=None, columns=None, rows=None, noise=None,
                 **kwargs):
        """
        A Thesaurus extension for storing feature vectors. Provides extra methods, e.g. dissect integration. Each
        entry can be of the form

            'water/N': [('nsubj-HEAD:title', 5), ('pobj-HEAD:by', 2)]

        i.e. entry: [(feature, count), ...]

        or of the form

        d = {
    '       monday': {
                'det:the': 23,
                'amod:awful': 1000,
                'amod:terrible': 243,
                ...
            },
    '       tuesday': { ... },
            ...
        }

        Difference between this class and Thesaurus:
         - removed allow_lexical_overlap and include_self parameters. It makes no sense to alter the features
         of an entry, but it is acceptable to pick and choose neighbours.
         - changed default value of sim_threshold to a very low value, for the same reason.
         - changed default value of merge_duplicates

        :param d: a dictionary that serves as a basis
        :param allow_lexical_overlap: if false, `get_nearest_neighbours` removes neighbours that have a unigram that is
         also the query entry. For example, `big_cat` won't be a neighbour of either `cat` or `big_dog`.
         NOTE: THE BEHAVIOUR OF THIS PARAMETER IS SLIGHTLY DIFFERENT FROM THE EQUIVALENT IN THESAURUS. This class
         compares strings, so `net/N` != `net/J` won't be neighbours, and Thesaurus compares `Token` objects,
         which currently do not take PoS tags into account, so `net/N` !== `net/J`.
        :param immutable: see Thesaurus docs
        :param matrix: can provide the data as a matrix, to avoid building it ourselves.
        :param noise: add uniform random noise to all non-zero entries in all vectors. The noise is in
        (-noise, noise). Because noise is only added to non-zero entries, this may only make sense
        for dense, low-dimensional vectors.
        """
        self._obj = d  # the underlying data dict. Do NOT RENAME!
        self.immutable = immutable
        self.allow_lexical_overlap = allow_lexical_overlap

        # the matrix representation of this object
        if matrix is None and columns is None and rows is None:
            self.matrix, self.columns, self.row_names = self.to_sparse_matrix()
        else:
            self.matrix, self.columns, self.row_names = matrix, columns, rows
        if self.matrix.shape != (len(self.row_names), len(self.columns)):
            logging.error('Vectors matrix has shape %r, but indices are of size %r, %r',
                          self.matrix.shape, len(self.row_names), len(self.columns))
        if noise:
            logging.info('Adding uniform noise [-{0}, +{0}] to non-zero vector dimensions'.format(noise))
            self.matrix.data += np.random.uniform(-noise, noise, self.matrix.data.shape)
        self.name2row = {feature: i for (i, feature) in enumerate(self.row_names)}

    @classmethod
    def from_tsv(cls, tsv_file, sim_threshold=-1e20,
                 lowercasing=False, ngram_separator='_',
                 row_filter=lambda x, y: True,
                 column_filter=lambda x: True,
                 max_len=50, max_neighbours=1e8,
                 merge_duplicates=True,
                 immutable=True, **kwargs):
        """
        Changes the default value of the sim_threshold parameter of super. Features can have any value, including
        negative (especially when working with neural embeddings).
        :rtype: Vectors
        """
        # For vectors disallowing lexical overlap does not make sense at construction time, but should be
        # implemented in get_nearest_neighbours. A Thesaurus can afford to do the filtering when reading the
        # ready-made thesaurus from disk.
        allow_lexical_overlap = kwargs.pop('allow_lexical_overlap', True)
        if is_hdf(tsv_file):
            import pandas as pd

            df = pd.read_hdf(tsv_file, 'matrix')
            logging.info('Found a DF of shape %r in HDF file %s', df.shape, tsv_file)
            # pytables doesn't like unicode values and replaces them with an empty string.
            # pandas doesn't like duplicate values in index
            # remove these, we don't want to work with them anyway
            df = df[df.index != '']
            row_filter_mask = [row_filter(f, DocumentFeature.from_string(f)) for f in df.index]
            df = df[row_filter_mask]
            logging.info('Dropped non-ascii rows and applied row filter. Shape is now %r', df.shape)
            return DenseVectors(df, immutable=immutable,
                                allow_lexical_overlap=allow_lexical_overlap,
                                **kwargs)

        th = Thesaurus.from_tsv(tsv_file, sim_threshold=sim_threshold,
                                ngram_separator=ngram_separator,
                                allow_lexical_overlap=True,
                                row_filter=row_filter, column_filter=column_filter,
                                max_len=max_len, max_neighbours=max_neighbours,
                                merge_duplicates=merge_duplicates,
                                **kwargs)

        # get underlying dict from thesaurus
        if not th._obj:
            raise ValueError('No entries left over after filtering')
        return Vectors(th._obj, immutable=immutable,
                       allow_lexical_overlap=allow_lexical_overlap, **kwargs)

    @classmethod
    def from_pandas_df(cls, df, **kwargs):
        d = df.T.to_dict()
        assert len(d) == len(df)
        for entry in d.keys():
            d[entry] = sorted(d[entry].items())
        return Vectors(d, matrix=csr_matrix(df.values), rows=df.index,
                       columns=df.columns, **kwargs)

    @classmethod
    def from_dict_of_dicts(cls, d, **kwargs):
        """
        Load Vectors stored as dictionary of dictionaries.
        :param d: Vectors of form
            d = {
                'monday': {
                    'det:the': 23,
                    'amod:awful': 1000,
                    'amod:terrible': 243,
                    ...
                },
                'tuesday': { ... },
                ...
            }
        """
        return Vectors(d=d)

    @classmethod
    def from_glove_model(cls, vector_file):
        """
        WARNING: `glove_python` is required to use this function!

        Load a GloVe vector model.
        :param vector_path: path to glove model
        :return: a `Vectors` object
        """
        from glove import Glove

        model = Glove.load_stanford(vector_file)
        vocab = model.dictionary.keys()

        vectors = {}

        dims = model.no_components  # vector dimensionality

        dimension_names = ['f%02d' % i for i in range(dims)]
        for word in vocab:
            vectors[word] = zip(dimension_names, model.word_vectors[model.dictionary[word]])

        return Vectors(vectors)

    @classmethod
    def from_word2vec_model(cls, vector_path):
        """
        WARNING: `gensim` is required to use this function!

        Load a word2vec vector model.
        :param vector_path: path to word2vec model
        :return: a `Vectors` object
        """
        from gensim.models.word2vec import Word2Vec

        model = Word2Vec.load_word2vec_format(vector_path, binary=vector_path.endswith('bin'))
        vocab = model.vocab.keys()

        vectors = {}

        dims = len(model[next(iter(vocab))])  # vector dimensionality

        dimension_names = ['f%02d' % i for i in range(dims)]
        for word in vocab:
            vectors[word] = zip(dimension_names, model[word])

        return Vectors(vectors)

    @classmethod
    def from_wort_model(cls, wort):
        """
        Initialise Vectors from an existing `wort` model.
        :param wort: The fitted `wort` model
        :param index: The `wort` index, mapping row indices to row names
        :param inverted_index: The `wort` inverted index, mapping row names to row indices
        :return: `discoutils` Vectors model
        """

        index = wort.get_index()
        X = wort.get_matrix()

        # index is already sorted (but inverted_index isn't)
        row_names = index.values()

        # Check if dim reduction has already been carried out:
        if (X.shape[0] != X.shape[1]): # dim reduction already done!
            columns = list(range(X.shape[1])) # columns are not interpretable in that case, so simply enumerate them
        else:
            columns = row_names # Still a square, symmetric matrix!

        return Vectors(d=wort.to_dict(), matrix=X, columns=columns, rows=row_names)

    def to_tsv(self, events_path, entries_path='', features_path='',
               entry_filter=lambda x: True, row_transform=lambda x: x,
               gzipped=False, enforce_word_entry_pos_format=True, dense_hd5=False):
        """
        Writes this thesaurus to Byblo-compatible file like the one it was most likely read from. In the
        process converts all entries to a DocumentFeature, so all entries must be parsable into one. May reorder the
        features of each entry.

        :param events_path: file to write to
        :param entry_filter: Called for every DocumentFeature that is an entry in this thesaurus. The vector will
         only be written if this callable return true
        :param row_transform: Callable, any transformation that might need to be done to each entry before converting
         it to a DocumentFeature. This is needed because some entries (e.g. african/J:amod-HEAD:leader) are not
         directly convertible (needs to be african/J_leader/N). Use this if the entries cannot be converted to
         DocumentFeature, e.g. if the data isn't PoS tagged.
         :param dense_hd5: if true, convert to a pandas `DataFrame` and write to a compressed HDF file. This is a 30%
          faster and produces 30% smaller files than using `gzipped`. This is only suitable for matrices with a small
          number of columns- this method enforces a hard limit of 1000.
          Requires PyTables and HDF5.
        :return: the file name
        """
        if enforce_word_entry_pos_format:
            rows = {i: DocumentFeature.from_string(row_transform(feat)) for (feat, i) in self.name2row.items()}
        else:
            rows = {i: feat for (feat, i) in self.name2row.items()}

        if dense_hd5 and len(self.columns) <= 1000:
            write_vectors_to_hdf(self.matrix, self.row_names, self.columns, events_path)
        else:
            write_vectors_to_disk(coo_matrix(self.matrix), rows, self.columns, events_path,
                                  features_path=features_path, entries_path=entries_path,
                                  entry_filter=entry_filter, gzipped=gzipped)
        return events_path

    def to_plain_txt(self, events_path, entries_path='', features_path=''):
        self.to_tsv(events_path, entries_path=entries_path, features_path=features_path,
                    gzipped=False, dense_hd5=False)

    def to_dissect_core_space(self):
        """
        Converts this object to a composes.semantic_space.space.Space
        :rtype: composes.semantic_space.space.Space
        """
        from composes.matrix.sparse_matrix import SparseMatrix
        from composes.semantic_space.space import Space

        mat, cols, rows = self.to_sparse_matrix()
        mat = SparseMatrix(mat)
        return Space(mat, rows, cols)

    def to_dissect_sparse_files(self, output_prefix, row_transform=None):
        """
        Converting to a dissect sparse matrix format. Writes out 3 files: columns, rows and matrix

        :param output_prefix: str
        :param row_transform:
        """
        with open('{0}.rows'.format(output_prefix), 'w') as outfile:
            for entry in self.keys():
                outfile.write('{}\n'.format(row_transform(entry) if row_transform else entry))

        with open('{0}.sm'.format(output_prefix), 'w') as outfile:
            for entry in self.keys():
                tmp_entry = row_transform(entry) if row_transform else entry
                for feature, count in self[entry]:
                    outfile.write('{} {} {}\n'.format(tmp_entry, feature, count))

        # write dissect columns file
        with open('{}.cols'.format(output_prefix), 'w') as outfile:
            for feature in sorted(set(self.columns)):
                outfile.write('{}\n'.format(feature))

    def get_vector(self, entry):
        """
        Returns a vector for the given entry. This differs from `self.__getitem__` in that it returns a sparse matrix
        instead of a list of (feature, count) tuples (in any order). The entries of the vectors are in the order of
        `self.columns`, which is sorted.
        :param entry: the entry
        :type entry: str or DocumentFeature
        :return: vector for the entry
        :rtype: scipy.sparse.csr_matrix, or None
        """
        if isinstance(entry, DocumentFeature):
            entry = str(entry)
        try:
            row = self.name2row[entry]
        except KeyError:
            return None  # no vector for this
        return self.matrix[row, :]

    def init_sims(self, vocab=None, n_neighbors=10, strategy='linear', knn='brute', nn_metric='l2'):
        """
        Prepares a mini thesaurus by placing all entries in `vocab` in a data structure. After that it is possible to
        get the nearest neighbours of an entry that this object has a vector for amongst all entries in `vocab`.

        :param vocab: which entries to include in thesaurus. If None, all entries that this object has a vector
        for are used
        :type vocab: iterable of str
        :param n_neighbors: how many neighbours to return when calling `get_nearest_neighbours`. Less neighbours may
        be returned if `self.allow_lexical_overlap` is false. By default this parameter is quite high so that
        after removing all lexically overlapping neighbours there would still be some left. Clients are free to slice
        further. Also, one less neighbour will be returned for an entry `E` if
        `len(vocab)==N and E in vocab and n_neighbours == N`
        :param strategy: how to find nearest neighbours. Linear is the standard implementation, anything
        """
        if not vocab:
            vocab = self.keys()

        # the pool out of which nearest neighbours will be sampled
        self.search_pool = set(foo for foo in vocab if foo in self.name2row)
        selected_rows = [self.name2row[foo] for foo in vocab if foo in self.name2row]

        if not selected_rows:
            raise ValueError('None of the vocabulary items in the labelled set have associated vectors')
        row2name = {v: k for k, v in self.name2row.items()}
        self.selected_row2name = {new: row2name[old] for new, old in enumerate(selected_rows)}
        if n_neighbors > len(selected_rows):
            logging.warning('You requested %d neighbours to be returned, but there are only %d. Truncating.',
                            n_neighbors, len(selected_rows))
            n_neighbors = len(selected_rows)
        self.n_neighbours = n_neighbors

        # todo BallTree/KDTree do not support cosine out of the box. algorithm='brute' is slower overall
        # for larger datasets. Tt's faster to build, O(1), and slower to query. If using euclidean as an
        # alternative, change 1-dist to dist in get_nearest_neighbour. Also, reduce the default value of
        # k from 200 to get another boost in performance
        X = self.matrix[selected_rows, :]

        # thomas 29.12.2015: see slack msg by miro, with cosine dists, this doesn't work
        if nn_metric == 'l2' and X.shape[1] < 1000:
            # if the matrix is smallish, make it dense and used KD Tree, it's 20-100x faster to query
            # and 4x slower to build
            if issparse(X):
                X = X.A
            knn = 'kd_tree'
        self.nn = NearestNeighbors(algorithm=knn,
                                   metric=nn_metric,
                                   n_neighbors=n_neighbors).fit(X)
        if strategy != 'linear':
            self.get_nearest_neighbours = self.get_nearest_neighbours_skipping
        self.get_nearest_neighbours.cache_clear()

    @lru_cache(maxsize=2 ** 16)
    def get_nearest_neighbours_linear(self, entry):
        """
        Get the nearest neighbours of `entry` amongst all entries that `init_sims` was called with. Resutls are
        sorted in order of increasing distance. The top neighbour will never be the entry itself (to match
        Byblo's behaviour)
        If there aren't any neighbours (either because we don't have a vector for the entry or because all
        neghbours overlap) an empty list is returned
        """
        if not hasattr(self, 'nn'):
            logging.warning('init_sims has not been called. Calling with default settings.')
            self.init_sims()
        if entry not in self:
            return []

        # if `entry` is contained in the list of neighbours, it will be popped and one less neighbour will be returned
        # so we need to ask for one extra neighbour, but without exceeding the number of available neighbours
        n_neigh = min(self.n_neighbours + (entry in self.search_pool), len(self.search_pool))
        v = self.get_vector(entry)
        distances, indices = self.nn.kneighbors(v.A if issparse(v) else v,
                                                n_neighbors=n_neigh)
        neigh = [(self.selected_row2name[indices[0][i]], distances[0][i]) for i in range(indices.shape[1])]
        if not self.allow_lexical_overlap:
            neigh = self.remove_overlapping_neighbours(entry, neigh)
        if neigh:
            # remove self as neigh, avoid popping an empty list
            # if there are identical vectors, self might not be the first neighbour- scan a bit further
            for i in range(min(3, len(neigh))):
                if neigh[i][0] == entry:
                    neigh.pop(i)
                    break
        return neigh[:self.n_neighbours]

    @lru_cache(maxsize=2 ** 16)
    def get_nearest_neighbours_skipping(self, entry):
        # accumulate neighbours by repeatedly calling get_nn_linear
        original_entry = entry
        result = []
        selected_neighbours = set([entry])
        for i in range(self.n_neighbours):
            neigh = self.get_nearest_neighbours_linear(entry)
            # do not jump back to where we came from
            neigh = [foo for foo in neigh if foo[0] not in selected_neighbours]
            if not self.allow_lexical_overlap:
                # this is needed if we want all neighbours returned to
                # not overlap with the original entry
                # without it something like this can happen:
                # black cat-> big dog-> black panther-> big cat
                # item I in this list does not overlap with item I-1, but may overlap with item 0
                # whether I want this is a different question
                neigh = self.remove_overlapping_neighbours(original_entry, neigh)
            if not neigh:
                break  # we are out of options
            entry = neigh[0][0]
            selected_neighbours.add(entry)
            result.append((entry, self.euclidean_distance(original_entry, entry)))
        return result

    get_nearest_neighbours = get_nearest_neighbours_linear

    @classmethod
    def from_shelf_readonly(cls, shelf_file_path, **kwargs):
        return Vectors(shelve.open(shelf_file_path, flag='r'), **kwargs)  # read only

    def _vector_distance(self, dist_fn, first, second):
        v1 = self.get_vector(first)
        v2 = self.get_vector(second)
        if v1 is not None and v2 is not None:
            return dist_fn( v1.A if issparse(v1) else v1,
                            v2.A if issparse(v2) else v2, )
        else:
            return None

    def euclidean_distance(self, first, second):
        return self._vector_distance(euclidean, first, second)

    def cosine_distance(self, first, second):
        return self._vector_distance(cosine, first, second)

    def __str__(self):
        return '[%d vectors]' % len(self)


class DenseVectors(Vectors):
    """
    A dense version of Vectors that stores data in a pandas DataFrame. This uses less
    memory for dense vectors and is much faster to read/write to disk.
    """

    def __init__(self, df, noise=False, **kwargs):
        self.df = df
        self.__dict__.update(**kwargs)

        self.matrix, self.columns, self.row_names = self.df.values, self.df.columns, self.df.index.values
        if noise:
            logging.info('Adding uniform noise [-{0}, +{0}] to non-zero vector dimensions'.format(noise))
            self.matrix += np.random.uniform(-noise, noise, self.matrix.shape)
        self.name2row = {feature: i for (i, feature) in enumerate(self.row_names)}

    def __contains__(self, item):
        if isinstance(item, DocumentFeature):
            item = str(item)
        return item in self.name2row

    def get_vector(self, item):
        if isinstance(item, DocumentFeature):
            item = str(item)
        if item not in self.name2row:
            return None
        return csr_matrix(self.df.ix[item].values)  # for compat with Vectors

    def __getitem__(self, item):
        return zip(self.columns, self.get_vector(item).A.ravel())

    def keys(self):
        return self.df.index

    def to_sparse_matrix(self):
        return csr_matrix(self.matrix), list(self.columns), list(self.row_names)

    def __len__(self):
        return len(self.row_names)

    def to_tsv(self, events_path, **kwargs):
        return super().to_tsv(events_path, dense_hd5=True)

    def to_plain_txt(self, events_path, entries_path='', features_path=''):
        super().to_tsv(events_path, entries_path=entries_path, features_path=features_path,
                       gzipped=False, dense_hd5=False)

    def __str__(self):
        return '[Dense vectors of shape {}]'.format(self.df.shape)


def as_plain_txt(path):
    v = Vectors.from_tsv(path)
    events_file = path + '.plain.txt'
    v.to_plain_txt(events_file)
    return events_file
