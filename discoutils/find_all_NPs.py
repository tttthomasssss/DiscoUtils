import argparse
import re
import logging
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from discoutils.misc import ContainsEverything

REPORTING_INTERVAL = 100000

noun_pattern = re.compile('^(\S+?/N)')  # a noun entry
adj_pattern = re.compile('^(\S+?/J)')  # an adjective entry
verb_pattern = re.compile('^(\S+?/V)')  # a verb entry

# an adjective modifier in an amod relation, consisting of non-whitespace
an_modifier_feature_pattern = re.compile('amod-DEP:(\S+/J)')

nn_modifier_feature_pattern = re.compile('nn-DEP:(\S+/N)')  # an noun modifier in a nn relation
nn_head_feature_pattern = re.compile('nn-HEAD:(\S+/N)')  # an noun head in a nn relation

grammatical_object_feature_pattern = re.compile("[di]obj-DEP:(\S+/N)")  # a verb and its corresponding object
grammatical_subject_feature_pattern = re.compile("nsubj-DEP:(\S+/N)")  # a verb and its corresponding nominal subject

window_feature_pattern = re.compile('(T:\S+)')  # an window feature, as output by FET

STOPWORDS = {'T:{}/'.format(x) for x in ENGLISH_STOP_WORDS}


def filter_features(features, *blacklist):
    """
    Removes stopword window features as well as ones mentioned in a list. Can be used to remove
    itself from features of phrases, e.g. black/J_cat/N might have T:black/J as a feature, which we
    do not want
    :param blacklist:
    :return:
    """
    # first remove the stopwords
    res = [f for f in features if not any(f.startswith(stopw) for stopw in STOPWORDS)]
    # now remove words in the blacklist
    blacklist = {'T:{}'.format(b) for b in blacklist}
    return [f for f in res if f not in blacklist]


def _get_NPs_in_line(line):
    noun_match = noun_pattern.match(line)
    if noun_match:
        head = noun_match.groups()[0]
        for pattern in [an_modifier_feature_pattern, nn_modifier_feature_pattern]:
            for modifier in pattern.findall(line):
                yield head, modifier


def _get_vp_in_line(line):
    obj = grammatical_object_feature_pattern.findall(line)  # match should work here but it doesnt. wtf?
    if obj:  # a verb's object is contained in this line
        verb_match = verb_pattern.match(line)  # find the verb
        if not verb_match:
            return
        verb = verb_match.group(0)
        # check if a subject is also present
        subj = grammatical_subject_feature_pattern.findall(line)
        if subj:
            return subj[0], verb, obj[0]
        else:
            return verb, obj[0]


def get_window_vectors_for_NPs(infile, outstream, whitelist=ContainsEverything()):
    """
    Outputs observed proximity vectors of some NPs. NPs are identified by a regex match, e.g.
    noun entry that has an amod feature, etc. Multiple features of the same entry are not merged.

    Only lines containing features for a noun are considered. That way double counting is avoided, e.g.
        full-time/J	amod-HEAD:tribunal/N    T:problem
        tribunal/N	amod-DEP:full-time/J	T:problem
    The feature T:problem is only output once as a feature of the noun compound

    :param infile: FET-formatted feature file
    :param outstream: where to write vectors
    :param whitelist: set of NPs to consider. Default: all that occur in the input file
    """
    with open(infile) as inf:
        for i, line in enumerate(inf):
            if i % REPORTING_INTERVAL == 0:
                logging.info('Done %d lines', i)

            # check if there are any NPs in this line
            nps = _get_NPs_in_line(line)
            if nps:
                features = window_feature_pattern.findall(line)
                for head, modifier in nps:
                    filtered_feats = filter_features(features, head, modifier)
                    phrase = '{}_{}'.format(modifier, head)
                    if filtered_feats and phrase in whitelist:
                        outstream.write('{}\t{}\n'.format(phrase, '\t'.join(filtered_feats)))


def get_NPs(infile, outstream, whitelist=ContainsEverything()):
    """
    Finds all adjective-noun and noun-noun compounds in a FET output file.
    Optionally accepts only these ANs/NNs whose modifiers are contained in a set.
    Requires features to have an associated PoS tag, see discoutils/tests/resources/exp10head.pbfiltered
    :param infile: file to read
    :param outstream: where to white newline-separate NPs
    :param whitelist: set of modifiers
    """
    with open(infile) as inf:
        for i, line in enumerate(inf):
            if i % REPORTING_INTERVAL == 0:
                logging.info('Done %d lines', i)
            for head, modifier in _get_NPs_in_line(line):
                if modifier in whitelist:
                    phrase = '{}_{}'.format(modifier, head)
                    outstream.write('%s\n' % phrase)


def get_VPs(infile, outstream, whitelist=ContainsEverything()):
    """
    Finds all verb phrases (verb-object and subject-verb-object compounds) in a FET output file.
    This only looks at lines that contain a grammatical object, and finds the verb and optional
    subject of the verb.
    :param whitelist: set of verbs. VPs whose head is not in that set are not output
    """
    with open(infile) as inf:
        for i, line in enumerate(inf):
            if i % REPORTING_INTERVAL == 0:
                logging.info('Done %d lines', i)

            vp = _get_vp_in_line(line)
            if vp:
                if vp[-2] not in whitelist:
                    # ignore verbs that are not in the whitelist
                    continue
                if len(vp) == 3:
                    # we found a SVO
                    phrase = '{}_{}_{}'.format(*vp)
                elif len(vp) == 2:
                    # just a verb-object compound
                    phrase = '{}_{}'.format(*vp)
                outstream.write('%s\n' % phrase)


def get_window_vectors_for_VPs(infile, outstream, whitelist=ContainsEverything()):
    """
    :param whitelist: set of VPs. Only VPs in that set are output
    """
    with open(infile) as inf:
        for i, line in enumerate(inf):
            if i % REPORTING_INTERVAL == 0:
                logging.info('Done %d lines', i)

            # check if there are any VPs in this line
            vp = _get_vp_in_line(line)
            if vp:
                features = filter_features(window_feature_pattern.findall(line), *vp)
                if len(vp) == 3:
                    # we found a SVO
                    phrase = '{}_{}_{}'.format(*vp)
                elif len(vp) == 2:
                    # just a verb-object compound
                    phrase = '{}_{}'.format(*vp)
                if features and phrase in whitelist:
                    outstream.write('{}\t{}\n'.format(phrase, '\t'.join(features)))


def read_configuration():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='Input file, as produced by FET')
    parser.add_argument('-o', '--output', required=False, type=str, default=None,
                        help='Name of output file. Default <input file>.ANsNNs')
    parser.add_argument('-s', '--whitelist', required=False, nargs='+',
                        help='Name of file containing a set of phrases/modifiers that will be considered. '
                             'All other NPs are disregarded. When vectors is true, these need to be phrases, '
                             'otherwise they need to be modifiers.')
    parser.add_argument('-v', '--vectors', action='store_true',
                        help='If set, will also output '
                             'window features for each entry occurrence. Default is False')
    return parser.parse_args()


def get_function_to_run(conf, use_VPs=False):
    if use_VPs:
        logging.info('Will only output a list of VPs')
    else:
        logging.info('Will only output a list of NPs')

    if conf.vectors:
        logging.info('Will also output window features for each entry occurrence')
        function = get_window_vectors_for_VPs if use_VPs else get_window_vectors_for_NPs
    else:
        function = get_VPs if use_VPs else get_NPs

    whitelist_files = conf.whitelist
    if whitelist_files:
        whitelist = set()
        for wfile in whitelist_files:
            with open(wfile) as f:
                whitelist |= set(x.strip() for x in f.readlines())
    else:
        whitelist = ContainsEverything()

    return function, whitelist


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s """
                               "(line %(lineno)d)\t%(levelname)s : %(""message)s")
    conf = read_configuration()
    output = conf.output if conf.output else '%s.ANsNNs' % conf.input

    # do the noun phrases first
    function, whitelist = get_function_to_run(conf)
    with open(output, 'w') as outstream:
        logging.info('Reading from %s', conf.input)
        logging.info('Writing to %s', output)
        function(conf.input, outstream, whitelist=whitelist)

    # and then the verb phrases
    function, whitelist = get_function_to_run(conf, use_VPs=True)
    with open(output, 'a') as outstream:
        logging.info('Reading from %s', conf.input)
        logging.info('Writing to %s', output)
        function(conf.input, outstream, whitelist=whitelist)