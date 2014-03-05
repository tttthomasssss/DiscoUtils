import argparse
import logging

__author__ = 'Julie'

import re, sys


def count(filename, pos_patterns):
    logging.info('Counting entry patterns: %s', [x.pattern for x in pos_patterns])
    counts = {} #dictionary to hold counts of items

    with open(filename, 'r') as instream:
        logging.info("Reading %s", filename)
        linesread = 0
        for line in instream:
            initial = line.split('\t')[0]
            if any(posPATT.match(initial) for posPATT in pos_patterns):
                current = counts.get(initial, 0)
                counts[initial] = current + 1
            linesread += 1
            if linesread % 10000 == 0:
                logging.info("Read %d lines", linesread)
    return counts, linesread


def filterline(fields, pattern):
    #filter out features in exclusion_list
    if pattern:
        fields = [f for f in fields if pattern.match(f)]
        return '%s\n' % ('\t'.join(fields)) if fields else ''
    else:
        return '%s\n' % ('\t'.join(fields))


def do_filtering(filename, outstream, threshold, pos_patterns, feature_pattern, counts, total_lines):
    logging.info("Rereading %s", filename)
    with open(filename, 'r') as instream:
        linesprocessed = 0
        for line in instream:
            line = line.rstrip()
            fields = line.split('\t')
            initial = fields.pop(0)
            if any(p.match(initial) for p in pos_patterns):
                if counts[initial] > threshold:
                    fields = filterline(fields, feature_pattern)
                    if fields:
                        outstream.write('%s\t%s' % (initial, fields))
            linesprocessed += 1
            if linesprocessed % 10000 == 0:
                percent = linesprocessed * 100. / total_lines
                logging.info("Processed %d lines (%2.1f percent)", linesprocessed, percent)


def read_configuration():
    #first 2 args must be filename and frequency threshold
    pos_patterns = {'N': re.compile('.*/N'),
                    'V': re.compile('.*/V'),
                    'J': re.compile('.*/J'),
                    'R': re.compile('.*/RB')}

    feature_patterns = {'wins': re.compile('T:.*'),
                        'deps': re.compile('.+-(HEAD|DEP):.+'),
                        'all': re.compile('.+'),
                        }

    def pos_pattern_validator(v):
        try:
            return pos_patterns[v]
        except:
            raise argparse.ArgumentTypeError("String '%s' does not match required format" % v)

    def feature_pattern_validator(v):
        try:
            return feature_patterns[v]
        except:
            raise argparse.ArgumentTypeError("String '%s' does not match required format" % v)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename', help='Input file')
    parser.add_argument('threshold', help='Entry frequency threshold', type=int, default=1)

    parser.add_argument('-pos', required=False, type=pos_pattern_validator, nargs='+',
                        default={pos_patterns['N']},
                        help='Entry type to accept. Valid choices are N, V, J, R')
    parser.add_argument('-feats', required=False, default='deps', type=feature_pattern_validator,
                        help='Feature type to accept. Valid choices are deps, wins or all')

    parser.add_argument('-o', '--output', required=False, default=None,
                        help='Name of output file. Default')

    return parser.parse_args()


def main():
    parameters = read_configuration()
    output = parameters.output if parameters.output else '%s.pbfiltered' % parameters.filename
    logging.info("Writing %s", output)

    counts, total_lines = count(parameters.filename, parameters.pos) # make count dictionary
    logging.info("Number of counted words is %d", len(counts))

    with open(output, 'w') as outfile:
        logging.info('Writing to %s', output)
        do_filtering(parameters.filename, outfile, parameters.threshold,
                     parameters.pos, parameters.feats, counts, total_lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s """
                               "(line %(lineno)d)\t%(levelname)s : %(""message)s")
    main()


