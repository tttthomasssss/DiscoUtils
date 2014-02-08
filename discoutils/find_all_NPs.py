import argparse
import re
import logging

__author__ = 'miroslavbatchkarov'

'''
Finds all adjective-noun and noun-noun compounds in a FET output file.
Optionally accepts only these ANs/NNs whose modifier is mentioned in a file.
Requires features to have an associated PoS tag, see
discoutils/tests/resources/exp10head.pbfiltered
'''


class ContainsEverything(object):
    def __contains__(self, item):
        return True



noun_pattern = re.compile('^(\S+?/N).+')  # a noun, non-greedy
an_pattern = re.compile('amod-DEP:(\S+?/J)')  # an adjective in an amod relation, consisting of non-whitespace

nn_pattern = re.compile('nn-DEP:(\S+?/N)')  # an noun in a nn relation


def go(infile, outstream, seed_set=ContainsEverything()):
    with open(infile) as inf:
        for i, line in enumerate(inf):
            if i % 100000 == 0:
                logging.info('Done %d lines', i)

            noun_match = noun_pattern.match(line)
            if noun_match:
                head = noun_match.groups()[0]
                for np_type, pattern in zip(['AN', 'NN'], [an_pattern, nn_pattern]):
                    for modifier in pattern.findall(line):
                        if modifier in seed_set:
                            outstream.write('{}:{}_{}\n'.format(np_type, modifier, head))


def read_configuration():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='Input file')
    parser.add_argument('-o', '--output', required=False, type=str, default=None,
                        help='Name of output file. Default <input file>.ANsNNs')
    parser.add_argument('-s', '--modifier_set', required=False, type=str, default='',
                        help='Name of file containing a set of modifiers that will be considered. '
                             'All other ANs/NNs are disregarded.')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s """
                               "(line %(lineno)d)\t%(levelname)s : %(""message)s")
    conf = read_configuration()
    output = conf.output if conf.output else '%s.ANsNNs' % conf.input

    seed = conf.modifier_set
    if seed:
        with open(seed) as f:
            seed = set(x.strip() for x in f.readlines())

    with open(output, 'w') as outstream:
        logging.info('Reading from %s', conf.input)
        logging.info('Writing to %s', output)
        if seed:
            go(conf.input, outstream, seed_set=seed)
        else:
            go(conf.input, outstream)
