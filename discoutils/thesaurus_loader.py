# coding=utf-8
from collections import Counter
import logging
import shelve
import numpy
from discoutils.tokens import DocumentFeature
from discoutils.collections_utils import walk_nonoverlapping_pairs
from discoutils.io_utils import write_vectors_to_disk
from discoutils.misc import ContainsEverything


class Thesaurus(object):
    def __init__(self, d):

        """
         A container that can read Byblo-formatted events (vectors) files OR sims files. Each entry can be of the form

            'water/N': [('nsubj-HEAD:title', 5), ('pobj-HEAD:by', 2)]

        i.e. entry: [(feature, count), ...], OR

            'water/N': [('horse/N', 0.5), ('earth/N', 0.4)]

        i.e. entry: [(neighbour, similarity), ...]

        :param d: a dictionary that serves as a basis
        """
        self.d = d

    def __getattr__(self, name):
        return getattr(self.d, name)

    def __setitem__(self, key, value):
        self.d[key] = value

    def __delitem__(self, key):
        del self.d[key]

    def __getitem__(self, item):
        return self.d[item]

    def __contains__(self, item):
        return item in self.d

    def __len__(self):
        return len(self.d)

    @classmethod
    def from_shelf_readonly(cls, shelf_file_path):
        return Thesaurus(shelve.open(shelf_file_path, flag='r'))  # read only

    @classmethod
    def from_tsv(cls, thesaurus_files='', sim_threshold=0, include_self=False,
                 lowercasing=False, ngram_separator='_', allow_lexical_overlap=True,
                 row_filter=lambda x, y: True, column_filter=lambda x: True, max_len=50,
                 max_neighbours=1e8):
        """
        Create a Thesaurus by parsing a Byblo-compatible TSV files (events or sims).
        If duplicate values are encoutered during parsing, only the latest will be kept.

        :param thesaurus_files: list or tuple of file paths to parse
        :type thesaurus_files: list
        :param sim_threshold: min count for inclusion in this object
        :type sim_threshold: float
        :param include_self: whether to include self as nearest neighbour. Only applicable when holding
         similarities and not vectors
        :type include_self: bool
        :param lowercasing: if true, most of what is read will be lowercased (excluding PoS tags), so
            Cat/N -> cat/N. This is desirable when reading full thesauri with this class. If False, no lowercasing
            will take place. This might be desirable when readings feature lists or already lowercased neighbour
            lists. FET + Byblo thesauri are already lowercased.
        :type lowercasing: bool
        :param ngram_separator: When n_gram entries are read in, what are the indidivual tokens separated by
        :param column_filter: A function that takes a string (column in the file) and returns whether or not
        the string should be kept
        :param row_filter: takes a string and its corresponding DocumentFeature and determines if it should be loaded
        :param allow_lexical_overlap: whether neighbours/features are allowed to overlap lexically with the entry
        they are neighbours/features of
        :param max_len: maximum length (in characters) of permissible entries. Longer entries are ignored.
        :param max_neighbours: maximum neighbours/features per entry. This is applied AFTER the filtering defined by
        column_filter and allow_lexical_overlap is finished.
        """

        if not thesaurus_files:
            logging.warn("No thesaurus specified")
            return {}

        to_return = dict()
        for path in thesaurus_files:
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
                        key = _smart_lower(tokens[0], ngram_separator, lowercasing)
                        dfkey = DocumentFeature.from_string(key)

                        if dfkey.type == 'EMPTY' or (not row_filter(key, dfkey)) or len(key) > max_len:
                            # do not load things in the wrong format, they'll get in the way later
                            continue

                        to_insert = [(_smart_lower(word, ngram_separator, lowercasing), float(sim))
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
                                #  has not already been lowercased- may result in
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

    def to_dissect_sparse_files(self, output_prefix, row_transform=None):
        """
        Converting to a dissect sparse matrix format. Writes out 3 files

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

    def to_sparse_matrix(self, row_transform=None, dtype=numpy.float):
        """
        Converts the vectors held in this object to a scipy sparse matrix
        :return: a tuple containing
            1) the sparse matrix, in which rows correspond to the order of this object's iteritems()
            2) a sorted list of all features (column labels of the matrix)
            3) a sorted list of all entries (row labels of the matrix)
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

    def to_dissect_core_space(self):
        """
        Converts this object to a composes.semantic_space.space.Space
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

    def to_tsv(self, filename, entry_filter=lambda x: True, row_transform=lambda x: x):
        """
        Writes this thesaurus to a Byblo-compatible events file like the one it was most likely read from. In the
        process converts all entries to a DocumentFeature.
        :param filename:
        :param entry_filter: Called for every DocumentFeature that is an entry in this thesaurus. The vector will
         only be written if this callable return true
        :param row_transform: Callable, any transformation that might need to be done to each entry before converting
         it to a DocumentFeature. This is needed because some entries (e.g. african/J:amod-HEAD:leader) are not
         directly convertible (needs to be african/J leader/N)
        :return: :rtype:
        """
        mat, cols, rows = self.to_sparse_matrix(row_transform=row_transform)
        rows = [DocumentFeature.from_string(x) for x in rows]
        write_vectors_to_disk(mat.tocoo(), rows, cols, filename, entry_filter=entry_filter)
        return filename


# END OF CLASS
def _smart_lower(words_with_pos, separator='_', lowercasing=True):
    """
    Lowercase just the words and not their PoS tags
    """
    if not lowercasing:
        return words_with_pos

    unigrams = words_with_pos.split(separator)
    words = []
    for unigram in unigrams:
        try:
            word, pos = unigram.split('/')
        except ValueError:
            # no pos
            word, pos = words_with_pos, ''

        words.append('/'.join([word.lower(), pos]) if pos else word.lower())

    return separator.join(words)
