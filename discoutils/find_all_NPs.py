import argparse
import re
import logging

__author__ = 'miroslavbatchkarov'

'''
Finds all adjective-noun and noun-noun compounds in a FET output file.
Requires features to have an associated PoS tag, see
discoutils/tests/resources/exp10head.pbfiltered
'''

noun_pattern = re.compile('^(.+?/N).+') # a noun, non-greedy
an_pattern = re.compile('amod-DEP:(.+?/J)') # an adjective in an amod relation

nn_pattern = re.compile('nn-DEP:(.+?/N)') # an noun in a nn relation


def go(infile, outstream):
    with open(infile) as inf:
        for i, line in enumerate(inf):
            if i % 10000 == 0:
                logging.info('Done %d lines', i)

            noun_match = noun_pattern.match(line)
            if noun_match:
                head = noun_match.groups()[0]
                for np_type, pattern in zip(['AN', 'NN'], [an_pattern, nn_pattern]):
                    for modifier in pattern.findall(line):
                        outstream.write('{}:{}_{}\n'.format(np_type, modifier, head))


def read_configuration():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='Input file')
    parser.add_argument('-o', '--output', required=False, default=None,
                        help='Name of output file. Default')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s """
                               "(line %(lineno)d)\t%(levelname)s : %(""message)s")
    conf = read_configuration()
    output = conf.output if conf.output else '%s.ANsNNs' % conf.input
    with open(output, 'w') as outstream:
        logging.info('Reading from %s', conf.input)
        logging.info('Writing to %s', output)
        go(conf.input, outstream)