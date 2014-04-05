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
        if isinstance(key, str):
            if isinstance(self.d, dict):
                # given a string to store into a dict. make that into a document feature
                key = DocumentFeature.from_string(key)
        else:
            if not isinstance(self.d, dict):
                # given a document feature to shelve, make it into a string (shelf needs a string)
                key = key.tokes_as_str()

        self.d[key] = value

    def __delitem__(self, key):
        if isinstance(key, str):
            key = DocumentFeature.from_string(key)
        del self.d[key]

    def __getitem__(self, item):
        if isinstance(item, str):
            neighbours = self.d[DocumentFeature.from_string(item)]
            return [(x[0].tokens_as_str(), x[1]) for x in neighbours]
        else:
            if isinstance(self.d, dict):
                # no shelving going on, look up directly
                return self.d[item]
            else:
                # shelve keys must be string
                return self.d[item.tokens_as_str()]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, item):
        if isinstance(item, str) and isinstance(self.d, dict):
            item = DocumentFeature.from_string(item)
        if isinstance(item, DocumentFeature) and not isinstance(self.d, dict):
            item = item.tokens_as_str()
        return item in self.d

    def __len__(self):
        return len(self.d)

    @classmethod
    def from_tsv(cls, thesaurus_files='', sim_threshold=0, include_self=False,
                 aggressive_lowercasing=True, ngram_separator='_', vocabulary=ContainsEverything(),
                 allow_lexical_overlap=True):
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
        :param aggressive_lowercasing: if true, most of what is read will be lowercased (excluding PoS tags), so
            Cat/N -> cat/N. This is desirable when reading full thesauri with this class. If False, no lowercasing
            will take place. This might be desirable when readings feature lists
        :type aggressive_lowercasing: bool
        :param ngram_separator: When n_gram entries are read in, what are the indidivual tokens separated by
        :param vocabulary: a set of DocumentFeature-s. Features not contained in this set will be discarded.
        :param allow_lexical_overlap: whether neighbours/features are allowed to overlap lexically with the entry
        they are neighbours/features of
        """
        return cls._read_from_disk(thesaurus_files, sim_threshold, include_self, ngram_separator,
                                   aggressive_lowercasing, vocabulary, allow_lexical_overlap)


    @classmethod
    def _read_from_disk(cls, thesaurus_files, sim_threshold, include_self, ngram_separator,
                        aggressive_lowercasing, vocabulary, allow_lexical_overlap):
        """
        Loads a set Byblo-generated thesaurus form the specified file and
        returns their union. If any of the files has been parsed already a
        cached version is used.

        Parameters:
        thesaurus_files: string, path the the Byblo-generated thesaurus
        use_pos: boolean, whether the PoS tags should be stripped from
        entities (if they are present)
        sim_threshold: what is the min similarity for neighbours that
        should be loaded

        Returns:
        A set of thesauri or an empty dictionary
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
                    entries = line.strip().split('\t')
                    if len(entries) % 2 == 0:
                        # must have an odd number of things, one for the entry
                        # and pairs for (neighbour, similarity)
                        logging.warn('Dodgy line in thesaurus file: %s\n %s', path, line)
                        continue
                    if entries[0] != FILTERED:
                        key = DocumentFeature.from_string(_smart_lower(entries[0],
                                                                       ngram_separator,
                                                                       aggressive_lowercasing))
                        if key.type == 'EMPTY':
                            # do not load things in the wrong format, they'll get in the way later
                            logging.info('Skipping thesaurus entry %s', key)
                            continue
                        to_insert = []
                        for (neighbour, sim) in walk_nonoverlapping_pairs(entries, 1):
                            text = _smart_lower(neighbour, ngram_separator, aggressive_lowercasing)
                            sim = float(sim)
                            if text.lower() != FILTERED and sim > sim_threshold:
                                # ensure filtered is removed
                                # ensure min similirity requirement is met
                                df = DocumentFeature.from_string(text)
                                if df.type == 'EMPTY':
                                    continue

                                if not allow_lexical_overlap and any(t in key.tokens for t in df.tokens):
                                    # ignore potentially lexically overlapping neighbours
                                    continue
                                if df not in vocabulary:
                                    # ignore neighbours not in the predefined vocabulary
                                    continue
                                to_insert.append((df, sim))

                        if include_self:
                            to_insert.insert(0, (key, 1.0))

                        # the steps above may filter out all neighbours of an entry. if this happens,
                        # do not bother adding it
                        if len(to_insert) > 0:
                            if key in to_return:
                                # todo this better not be a neighbours file, merging doesn't work there
                                logging.warn('Multiple entries for "%s" found. Merging.' % entries[0])
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
        d = shelve.open(filename, flag='c', writeback=True)  # read and write
        for entry, features in self.iteritems():
            d[entry.tokens_as_str()] = features
        d.close()

    def to_dissect_sparse_files(self, output_prefix, row_transform=None, column_transform=None):
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
                    outfile.write('{} {} {}\n'.format(tmp_entry,
                                                      column_transform(feature) if column_transform else feature,
                                                      count))

        # write dissect columns file
        columns = set(feature for vector in self.values() for (feature, count) in vector)
        with open('{}.cols'.format(output_prefix), 'w+b') as outfile:
            for feature in sorted(columns):
                outfile.write('{}\n'.format(column_transform(feature) if column_transform else feature))

    def to_sparse_matrix(self, row_transform=DocumentFeature.tokens_as_str,
                         column_transform=DocumentFeature.tokens_as_str, dtype=numpy.float):
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
        cols = self.v.feature_names_
        if column_transform:
            cols = map(column_transform, cols)

        return mat, cols, rows

    def to_dissect_core_space(self):
        """
        Converts this object to a composes.semantic_space.space.Space
        """
        from composes.matrix.sparse_matrix import SparseMatrix
        from composes.semantic_space.space import Space

        mat, cols, rows = self.to_sparse_matrix(row_transform=DocumentFeature.tokens_as_str,
                                                column_transform=DocumentFeature.tokens_as_str)
        mat = SparseMatrix(mat)
        s = Space(mat, rows, cols)

        # the dimensions of the semantic space are stored as DocumentFeature-s in self
        # they need to be output as strings
        old_voc = self.v.vocabulary_
        transformed_voc = {DocumentFeature.tokens_as_str(k): v for k, v in old_voc.iteritems()}
        self.v.vocabulary_ = transformed_voc

        # test that the mapping from string to its vector has not been messed up
        for i in range(min(10, len(self))):
            s1 = s.get_row(rows[i]).mat
            s2 = self.v.transform(dict(self[rows[i]]))
            # sparse matrices do not currently support equality testing
            assert abs(s1 - s2).nnz == 0
        self.v.vocabulary_ = old_voc
        return s

    def to_file(self, filename, entry_filter=lambda x: True,
                row_transform=DocumentFeature.tokens_as_str,
                column_transform=DocumentFeature.tokens_as_str):
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
        mat, cols, rows = self.to_sparse_matrix(row_transform=None, column_transform=None)
        write_vectors_to_disk(mat.tocoo(), rows, cols, filename, entry_filter=entry_filter,
                              row_transform=row_transform, column_transform=column_transform)
        return filename


# END OF CLASS
def _smart_lower(words_with_pos, separator='_', aggressive_lowercasing=True):
    """
    Lowercase just the words and not their PoS tags
    """
    if not aggressive_lowercasing:
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
