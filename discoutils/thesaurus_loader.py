# coding=utf-8
from collections import Counter
import logging
import shelve
import numpy
from discoutils.tokens import DocumentFeature
import six
from discoutils.collections_utils import walk_nonoverlapping_pairs
from discoutils.io_utils import write_vectors_to_disk
from sklearn.neighbors import NearestNeighbors

try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache  # py2


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
        to_insert = [(f[0].tokens_as_str(), f[1]) for f in features
                     if not any(t in entry.tokens for t in f[0].tokens)]
        return to_insert

    @classmethod
    def from_tsv(cls, tsv_file, sim_threshold=0, include_self=False,
                 lowercasing=False, ngram_separator='_', allow_lexical_overlap=True,
                 row_filter=lambda x, y: True, column_filter=lambda x: True, max_len=50,
                 max_neighbours=1e8, merge_duplicates=False, immutable=True,
                 enforce_word_entry_pos_format=True, **kwargs):
        """
        Create a Thesaurus by parsing a Byblo-compatible TSV files (events or sims).
        If duplicate values are encoutered during parsing, only the latest will be kept.

        :param tsv_file: path to input TSV file
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

        to_return = dict()
        logging.info('Loading thesaurus %s from disk', tsv_file)
        if not allow_lexical_overlap:
            logging.warning('DISALLOWING LEXICAL OVERLAP')

        if not allow_lexical_overlap and not enforce_word_entry_pos_format:
            raise ValueError('allow_lexical_overlap requires entries to be converted to a DocumentFeature. '
                             'Please enable enforce_word_entry_pos_format')
        FILTERED = '___FILTERED___'.lower()
        with open(tsv_file) as infile:
            for line in infile:
                tokens = line.strip().split('\t')
                if len(tokens) % 2 == 0:
                    # must have an odd number of things, one for the entry
                    # and pairs for (neighbour, similarity)
                    logging.warning('Skipping dodgy line in thesaurus file: %s\n %s', tsv_file, line)
                    continue

                if tokens[0] != FILTERED:
                    key = DocumentFeature.smart_lower(tokens[0], ngram_separator, lowercasing)
                    dfkey = DocumentFeature.from_string(key) if enforce_word_entry_pos_format else None

                    if enforce_word_entry_pos_format and dfkey.type == 'EMPTY':
                        # do not load things in the wrong format, they'll get in the way later
                        logging.warning('%s is not in the word/POS format, skipping', tokens[0])
                        continue

                    if (not row_filter(key, dfkey)) or len(key) > max_len:
                        logging.warning('Skipping entry for %s', key)
                        continue

                    to_insert = [(DocumentFeature.smart_lower(word, ngram_separator, lowercasing), float(sim))
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
                                logging.warning('Multiple entries for "%s" found. Merging.', tokens[0])
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

    def to_tsv(self, filename):
        """
        Writes this thesaurus to a Byblo-compatible sims file like the one it was most likely read from.  Neighbours
        are written in the order that they appear in.
        :param filename: file to write to
        :return: the file name
        """
        logging.warning('row_transform and entry_filter options are ignored in order to use preserve_order')
        with open(filename, 'w') as outfile:
            for entry, vector in self._obj.items():
                features_str = '\t'.join(['%s\t%f' % foo for foo in vector])
                outfile.write('%s\t%s\n' % (entry, features_str))
        return filename

    def to_sparse_matrix(self, row_transform=None, dtype=numpy.float):
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

    def __getattr__(self, name):
        return getattr(self._obj, name)

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
            key = DocumentFeature.tokens_as_str(key)
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
            item = DocumentFeature.tokens_as_str(key)

        del self._obj[key]
        if hasattr(self, 'matrix'):
            mask = numpy.ones(self.matrix.shape[0], dtype=bool)
            mask[self.name2row[key]] = False
            self.matrix = self.matrix[mask, :]

    def __getitem__(self, item):
        if isinstance(item, DocumentFeature):
            item = DocumentFeature.tokens_as_str(item)
        return self._obj[item]

    get_nearest_neighbours = __getitem__

    def __contains__(self, item):
        if isinstance(item, DocumentFeature):
            item = DocumentFeature.tokens_as_str(item)
        return item in self._obj

    def __len__(self):
        if not hasattr(self, 'cached_len'):
            # if self._obj is a shelve object, calling its __len__ takes forever
            self.cached_len = len(self._obj)
        return self.cached_len


class Vectors(Thesaurus):
    def __init__(self, d, immutable=True, allow_lexical_overlap=True, **kwargs):
        """
        A Thesaurus extension for storing feature vectors. Provides extra methods, e.g. dissect integration. Each
        entry can be of the form

            'water/N': [('nsubj-HEAD:title', 5), ('pobj-HEAD:by', 2)]

        i.e. entry: [(feature, count), ...]

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
        """
        self._obj = d  # the underlying data dict. Do NOT RENAME!
        self.immutable = immutable
        self.allow_lexical_overlap = allow_lexical_overlap

        # the matrix representation of this object
        self.matrix, self.columns, self.row_names = self.to_sparse_matrix()
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
        th = Thesaurus.from_tsv(tsv_file, sim_threshold=sim_threshold,
                                ngram_separator=ngram_separator,
                                allow_lexical_overlap=True,
                                row_filter=row_filter, column_filter=column_filter,
                                max_len=max_len, max_neighbours=max_neighbours,
                                merge_duplicates=merge_duplicates, **kwargs)

        # get underlying dict from thesaurus
        if not th._obj:
            raise ValueError('No entries left over after filtering')
        return Vectors(th._obj, immutable=immutable,
                       allow_lexical_overlap=allow_lexical_overlap)

    def to_tsv(self, events_path, entries_path='', features_path=''):
        """
        Writes this thesaurus to Byblo-compatible file like the one it was most likely read from. In the
        process converts all entries to a DocumentFeature, so all entries must be parsable into one. May reorder the
        features of each entry.

        :param events_file: file to write to
        :param entry_filter: Called for every DocumentFeature that is an entry in this thesaurus. The vector will
         only be written if this callable return true
        :param row_transform: Callable, any transformation that might need to be done to each entry before converting
         it to a DocumentFeature. This is needed because some entries (e.g. african/J:amod-HEAD:leader) are not
         directly convertible (needs to be african/J_leader/N)
        :return: the file name
        """
        rows = {i: feat for (feat, i) in self.name2row.items()}
        write_vectors_to_disk(self.matrix.tocoo(), rows, self.columns, events_path,
                              features_path=features_path, entries_path=entries_path)
        return events_path

    def to_dissect_core_space(self):
        """
        Converts this object to a composes.semantic_space.space.Space
        :rtype: composes.semantic_space.space.Space
        """
        from composes.matrix.sparse_matrix import SparseMatrix
        from composes.semantic_space.space import Space

        mat, cols, rows = self.to_sparse_matrix()
        mat = SparseMatrix(mat)
        s = Space(mat, rows, cols)

        # test that the mapping from string to its vector has not been messed up
        for i in range(min(10, len(self))):
            s1 = s.get_row(rows[i]).mat
            s2 = self.v.transform(dict(self[rows[i]]))
            # sparse matrices do not currently support equality testing
            assert abs(s1 - s2).nnz == 0

        return s

    def to_dissect_sparse_files(self, output_prefix, row_transform=None):
        """
        Converting to a dissect sparse matrix format. Writes out 3 files, columns, rows and matrix

        :param output_prefix: str, a
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
        columns = set(feature for vector in self.values() for (feature, count) in vector)
        with open('{}.cols'.format(output_prefix), 'w') as outfile:
            for feature in sorted(columns):
                outfile.write('{}\n'.format(feature))

    def get_vector(self, entry):
        """
        Returns a vector for the given entry. This differs from self.__getitem__ in that it returns a sparse matrix
        instead of a list of (feature, count) tuples
        :param entry: the entry
        :type entry: str
        :return: vector for the entry
        :rtype: scipy.sparse.csr_matrix, or None
        """
        try:
            row = self.name2row[entry]
        except KeyError:
            return None  # no vector for this
        return self.matrix[row, :]

    def init_sims(self, vocab=None, n_neighbors=200):
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

        # todo BallTree/KDTree do not support cosine out of the box. algorithm='brute' may be slower overall
        # it's faster to build, O(1), and slower to query
        self.nn = NearestNeighbors(algorithm='brute',
                                   metric='cosine',
                                   n_neighbors=n_neighbors).fit(self.matrix[selected_rows, :])
        self.get_nearest_neighbours.cache_clear()

    @lru_cache(maxsize=2 ** 13)
    def get_nearest_neighbours(self, entry):
        """
        Get the nearest neighbours of `entry` amongst all entries that `init_sims` was called with. The top
        neighbour will never be the entry itself (to match Byblo's behaviour)
        """
        if isinstance(entry, DocumentFeature):
            entry = entry.tokens_as_str()
        if not hasattr(self, 'nn'):
            logging.warning('init_sims has not been called. Calling with default settings.')
            self.init_sims()
        if entry not in self:
            return None

        # if `entry` is contained in the list of neighbours, it will be popped and one less neighbour will be returned
        # so we need to ask for one extra neighbour, but without exceeding the number of available neighbours
        n_neigh = min(self.n_neighbours + (entry in self.search_pool), len(self.search_pool))
        distances, indices = self.nn.kneighbors(self.get_vector(entry),
                                                n_neighbors=n_neigh)
        neigh = [(self.selected_row2name[indices[0][i]], 1 - distances[0][i]) for i in range(indices.shape[1])]
        if not self.allow_lexical_overlap:
            neigh = self.remove_overlapping_neighbours(entry, neigh)
        if neigh[0][0] == entry:
            neigh.pop(0)
        return neigh[:self.n_neighbours]

    @classmethod
    def from_shelf_readonly(cls, shelf_file_path, **kwargs):
        return Vectors(shelve.open(shelf_file_path, flag='r'), **kwargs)  # read only

    def __str__(self):
        return '[%d vectors]' % len(self)

