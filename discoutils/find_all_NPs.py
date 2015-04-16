import argparse
import re
import logging
from discoutils.misc import ContainsEverything

noun_pattern = re.compile('^(\S+?/N)')  # a noun entry
adj_pattern = re.compile('^(\S+?/J)')  # an adjective entry

# an adjective modifier in an amod relation, consisting of non-whitespace
an_modifier_feature_pattern = re.compile('amod-DEP:(\S+/J)')
an_head_feature_pattern = re.compile('amod-HEAD:(\S+/N)')  # an adjective modifier/head in an amod relation

nn_modifier_feature_pattern = re.compile('nn-DEP:(\S+/N)')  # an noun modifier in a nn relation
nn_head_feature_pattern = re.compile('nn-HEAD:(\S+/N)')  # an noun head in a nn relation

window_feature_pattern = re.compile('(T:\S+)')  # an noun in a nn relation


def _find_and_output_features(second, line, first, np_type, outstream, phrase_set):
    if np_type and second and first:
        # if we've found en entry having the right PoS and dep features, get its window features
        features = window_feature_pattern.findall(line)
        phrase = '{}_{}'.format(first, second)
        if features and phrase in phrase_set:
            outstream.write('{}\t{}\n'.format(phrase, '\t'.join(features)))


def get_window_vectors(infile, outstream, whitelist=ContainsEverything()):
    """
    Outputs observed proximity vectors of some NPs. NPs are identified by a regex match, e.g.
    noun entry that has an amod feature, etc. Multiple features of the same entry are not merged.
    :param infile: FET-formatted feature file
    :param outstream: where to write vectors
    :param whitelist: set of NPs to consider. Default: all that occur in the input file
    """
    with open(infile) as inf:
        for i, line in enumerate(inf):
            if i % 100000 == 0:
                logging.info('Done %d lines', i)

            np_type, second, first = None, None, None
            # check if the entry is an adj or a noun
            nouns = noun_pattern.findall(line)
            adjectives = adj_pattern.findall(line)
            amod_heads = an_head_feature_pattern.findall(line)
            amod_modifiers = an_modifier_feature_pattern.findall(line)
            nn_heads = nn_head_feature_pattern.findall(line)
            nn_modifiers = nn_modifier_feature_pattern.findall(line)

            if nouns:
                assert 1 == len(nouns)  # no more than 1 entry per line
            if adjectives:
                assert 1 == len(adjectives)  # no more than 1 entry per line
            if adjectives and amod_modifiers:
                # logging.warning('Adjective has adjectival modifiers')
                continue

            if adjectives and amod_heads:
                first = adjectives[0]
                second = amod_heads[0]
                np_type = 'AN'
                _find_and_output_features(second, line, first, np_type, outstream, whitelist)

            if nouns and amod_modifiers:
                second = nouns[0]
                np_type = 'AN'
                for first in an_modifier_feature_pattern.findall(line):
                    _find_and_output_features(second, line, first, np_type, outstream, whitelist)

            if nouns and nn_heads:
                np_type = 'NN'
                first = nouns[0]
                second = nn_heads[0]
                _find_and_output_features(second, line, first, np_type, outstream, whitelist)

            if nouns and nn_modifiers:
                np_type = 'NN'
                second = nouns[0]
                for first in nn_modifier_feature_pattern.findall(line):
                    _find_and_output_features(second, line, first, np_type, outstream, whitelist)


def get_NPs(infile, outstream, whitelist=ContainsEverything()):
    """
    Finds all adjective-noun and noun-noun compounds in a FET output file.
    Optionally accepts only these ANs/NNs whose modifiers are contained in a set
    Requires features to have an associated PoS tag, see discoutils/tests/resources/exp10head.pbfiltered
    :param infile: file to read
    :param outstream: where to white newline-separate NPs
    :param whitelist: set of modifiers
    """
    with open(infile) as inf:
        for i, line in enumerate(inf):
            if i % 100000 == 0:
                logging.info('Done %d lines', i)

            noun_match = noun_pattern.match(line)
            if noun_match:
                head = noun_match.groups()[0]
                for pattern in [an_modifier_feature_pattern, nn_modifier_feature_pattern]:
                    for modifier in pattern.findall(line):
                        phrase = '{}_{}'.format(modifier, head)
                        if modifier in whitelist:
                            outstream.write('%s\n' % phrase)


def read_configuration():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='Input file, as produced by FET')
    parser.add_argument('-o', '--output', required=False, type=str, default=None,
                        help='Name of output file. Default <input file>.ANsNNs')
    parser.add_argument('-s', '--whitelist', required=False, type=str, default='',
                        help='Name of file containing a set of phrases/modifiers that will be considered. '
                             'All other NPs are disregarded. When vectors is true, these need to be phrases, '
                             'otherwise they need to be modifiers.')
    parser.add_argument('-v', '--vectors', action='store_true',
                        help='If set, will also output '
                             'window features for each entry occurrence. Default is False')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s """
                               "(line %(lineno)d)\t%(levelname)s : %(""message)s")
    conf = read_configuration()
    output = conf.output if conf.output else '%s.ANsNNs' % conf.input

    if conf.vectors:
        logging.info('Will also output window features for each entry occurrence')
        function = get_window_vectors
    else:
        logging.info('Will only output NP entries')
        function = get_NPs
    whitelist = conf.whitelist

    if whitelist:
        with open(whitelist) as f:
            whitelist = set(x.strip() for x in f.readlines())
    else:
        whitelist = ContainsEverything()

    with open(output, 'w') as outstream:
        logging.info('Reading from %s', conf.input)
        logging.info('Writing to %s', output)

        function(conf.input, outstream, whitelist=whitelist)