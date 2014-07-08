# coding=utf-8
from collections import Counter
import logging
import shelve
import numpy
from discoutils.tokens import DocumentFeature
from discoutils.collections_utils import walk_nonoverlapping_pairs
from discoutils.io_utils import write_vectors_to_disk


class Thesaurus(object):
    def __init__(self, d):

        """
         A container that can read Byblo-formatted  sims files. Each entry can be of the form

            'water/N': [('horse/N', 0.5), ('earth/N', 0.4)]

        i.e. entry: [(neighbour, similarity), ...]

        Use `Vectors` to store feature vectors

        :param d: a dictionary that serves as a basis. All magic method call on `Thesaurus` objects are forwarded
        to this dict so users can think of this class as a normal dict.
        """
        self._obj = d  # do not rename


    @classmethod
    def from_shelf_readonly(cls, shelf_file_path):
        return Thesaurus(shelve.open(shelf_file_path, flag='r'))  # read only

    @classmethod
    def from_tsv(cls, tsv_files='', sim_threshold=0, include_self=False,
                 lowercasing=False, ngram_separator='_', allow_lexical_overlap=True,
                 row_filter=lambda x, y: True, column_filter=lambda x: True, max_len=50,
                 max_neighbours=1e8):
        """
        Create a Thesaurus by parsing a Byblo-compatible TSV files (events or sims).
        If duplicate values are encoutered during parsing, only the latest will be kept.

        :param tsv_files: list or tuple of file paths to parse
        :type tsv_files: list
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
        :param row_filter: takes a string and its corresponding DocumentFeature and determines if it should be loaded
        :param allow_lexical_overlap: whether neighbours/features are allowed to overlap lexically with the entry
        they are neighbours/features of
        :param max_len: maximum length (in characters) of permissible **entries**. Longer entries are ignored.
        :param max_neighbours: maximum neighbours per entry. This is applied AFTER the filtering defined by
        column_filter and allow_lexical_overlap is finished.
        """

        if not tsv_files:
            logging.warn("No thesaurus specified")
            return {}

        to_return = dict()
        for path in tsv_files:
            logging.info('Loading thesaurus %s from disk', path)
            if not allow_lexical_overlap:
                logging.warn('DISALLOWING LEXICAL OVERLAP')

            FILTERED = '___FILTERED___'.lower()
            with open(path) as infile:
                for line in infile:
                    tokens = line.strip().split('\t')
                    if len(tokens) % 2 == 0:
                        # must have an odd number of things, one for the entry
                        # and pairs for (neighbour, similarity)
                        logging.warn('Dodgy line in thesaurus file: %s\n %s', path, line)
                        continue

                    if tokens[0] != FILTERED:
                        key = DocumentFeature.smart_lower(tokens[0], ngram_separator, lowercasing)
                        dfkey = DocumentFeature.from_string(key)

                        if dfkey.type == 'EMPTY' or (not row_filter(key, dfkey)) or len(key) > max_len:
                            # do not load things in the wrong format, they'll get in the way later
                            continue

                        to_insert = [(DocumentFeature.smart_lower(word, ngram_separator, lowercasing), float(sim))
                                     for (word, sim) in walk_nonoverlapping_pairs(tokens, 1)
                                     if word.lower() != FILTERED and column_filter(word) and float(sim) > sim_threshold]

                        if not allow_lexical_overlap:
                            features = [(DocumentFeature.from_string(x[0]), x[1]) for x in to_insert]
                            key_tokens = dfkey.tokens
                            to_insert = [(f[0].tokens_as_str(), f[1]) for f in features
                                         if not any(t in key_tokens for t in f[0].tokens)]

                        if len(to_insert) > max_neighbours:
                            to_insert = to_insert[:max_neighbours]

                        if include_self:
                            to_insert.insert(0, (key, 1.0))

                        # the steps above may filter out all neighbours of an entry. if this happens,
                        # do not bother adding it
                        if len(to_insert) > 0:

                            if key in to_return:
                                # todo this better not be a neighbours file, merging doesn't work there
                                logging.warn('Multiple entries for "%s" found. Merging.' % tokens[0])
                                c = Counter(dict(to_return[key]))
                                c.update(dict(to_insert))
                                to_return[key] = [(k, v) for k, v in c.iteritems()]
                            else:
                                to_return[key] = to_insert

                                # note- do not attempt to lowercase if the thesaurus
                                # has not already been lowercased- may result in
                                # multiple neighbour lists for the same entry
        return Thesaurus(to_return)

    def to_shelf(self, filename):
        """
        Uses the shelf module to persist this object to a file
        """
        logging.info('Shelving thesaurus of size %d to %s', len(self), filename)
        d = shelve.open(filename, flag='c')  # read and write
        for entry, features in self.iteritems():
            d[str(entry)] = features
        d.close()

    def to_tsv(self, filename):
        """
        Writes this thesaurus to a Byblo-compatible sims file like the one it was most likely read from.  Neighbours
        are written in the order that they appear in.
        :param filename: file to write to
        :return: the file name
        """
        logging.warn('row_transform and entry_filter options are ignored in order to use preserve_order')
        with open(filename, 'w') as outfile:
            for entry, vector in self._obj.iteritems():
                features_str = '\t'.join(['%s\t%f' % foo for foo in vector])
                outfile.write('%s\t%s\n' % (entry, features_str))
        return filename

    def to_sparse_matrix(self, row_transform=None, dtype=numpy.float):
        """
        Converts the vectors held in this object to a scipy sparse matrix.

        WARNING: This method doesn't make much sense for a Thesaurus and belongs in Vectors. I'm leaving it
        here as some existing Thesaurus tests rely on it.

        :return: a tuple containing
            1) the sparse matrix, in which rows correspond to the order of this object's iteritems()
            2) a **sorted** list of all features (column labels of the matrix).
            3) a list of all entries (row labels of the matrix)
        :rtype: tuple
        """
        from sklearn.feature_extraction import DictVectorizer

        self.v = DictVectorizer(sparse=True, dtype=dtype)

        # order in which rows are iterated is not guaranteed if the dict if modified, but we're not doing that,
        # so it's all fine
        mat = self.v.fit_transform([dict(fv) for fv in self.itervalues()])
        rows = [k for k in self.iterkeys()]
        if row_transform:
            rows = map(row_transform, rows)

        return mat, self.v.feature_names_, rows

    def __getattr__(self, name):
        return getattr(self._obj, name)

    def __setitem__(self, key, value):
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
            mask[self.rows[key]] = False
            self.matrix = self.matrix[mask, :]

    def __getitem__(self, item):
        if isinstance(item, DocumentFeature):
            item = DocumentFeature.tokens_as_str(item)
        return self._obj[item]

    def __contains__(self, item):
        if isinstance(item, DocumentFeature):
            item = DocumentFeature.tokens_as_str(item)
        return item in self._obj

    def __len__(self):
        return len(self._obj)

class Vectors(Thesaurus):
    def __init__(self, d):
        """
        A Thesaurus extension for storing feature vectors. Provides extra methods, e.g. dissect integration. Each
        entry can be of the form

            'water/N': [('nsubj-HEAD:title', 5), ('pobj-HEAD:by', 2)]

        i.e. entry: [(feature, count), ...]

        Difference between this class and Thesaurus:
         - removed allow_lexical_overlap and include_self parameters. It makes no sense to alter the features
         of an entry, but it is acceptable to pick and choose neighbours.
         - changed default value of sim_threshold to a very low value, for the same reason.

        :param d: a dictionary that serves as a basis
        """
        self._obj = d  # the underlying data dict. Do NOT RENAME!
        # the matrix representation of this object
        self.matrix, self.columns, rows = self.to_sparse_matrix()
        self.rows = {feature: i for (i, feature) in enumerate(rows)}

    @classmethod
    def from_tsv(cls, tsv_files='', sim_threshold=-1e20,
                 lowercasing=False, ngram_separator='_',
                 row_filter=lambda x, y: True,
                 column_filter=lambda x: True,
                 max_len=50, max_neighbours=1e8):
        """
        Changes the default value of the sim_threshold parameter of super. Features can have any value, including
        negative (especially when working with neural embeddings).
        :rtype: Vectors
        """
        th = Thesaurus.from_tsv(tsv_files=tsv_files, sim_threshold=sim_threshold,
                                ngram_separator=ngram_separator, allow_lexical_overlap=True,
                                row_filter=row_filter, column_filter=column_filter,
                                max_len=max_len, max_neighbours=max_neighbours)
        return Vectors(th._obj)  # get underlying dict from thesaurus

    def to_tsv(self, events_path, entries_path='', features_path='',
               entry_filter=lambda x: True, row_transform=lambda x: x):
        """
        Writes this thesaurus to Byblo-compatible file like the one it was most likely read from. In the
        process converts all entries to a DocumentFeature, so all entries must be parsable into one. May reorder the
        features of each entry.

        :param events_file: file to write to
        :param entry_filter: Called for every DocumentFeature that is an entry in this thesaurus. The vector will
         only be written if this callable return true
        :param row_transform: Callable, any transformation that might need to be done to each entry before converting
         it to a DocumentFeature. This is needed because some entries (e.g. african/J:amod-HEAD:leader) are not
         directly convertible (needs to be african/J leader/N)
        :return: the file name
        """
        # todo converting to a DocumentFeature is silly as any odd entry breaks this method
        logging.warn('Not attempting to preserve order of features when saving to TSV')
        # mat, cols, rows = self.to_sparse_matrix(row_transform=row_transform)
        rows = [DocumentFeature.from_string(x) for x in self.rows]
        write_vectors_to_disk(self.matrix.tocoo(), rows, self.columns, events_path,
                              features_path=features_path, entries_path=entries_path,
                              entry_filter=entry_filter)
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
        with open('{0}.rows'.format(output_prefix), 'w+b') as outfile:
            for entry in self.keys():
                outfile.write('{}\n'.format(row_transform(entry) if row_transform else entry))

        with open('{0}.sm'.format(output_prefix), 'w+b') as outfile:
            for entry in self.keys():
                tmp_entry = row_transform(entry) if row_transform else entry
                for feature, count in self[entry]:
                    outfile.write('{} {} {}\n'.format(tmp_entry, feature, count))

        # write dissect columns file
        columns = set(feature for vector in self.values() for (feature, count) in vector)
        with open('{}.cols'.format(output_prefix), 'w+b') as outfile:
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
            row = self.rows[entry]
        except KeyError:
            return None  # no vector for this
        return self.matrix[row, :]


    def __str__(self):
        return '[%d vectors]' % len(self)

