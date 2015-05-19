import argparse
import logging
import os
import sys
from discoutils.misc import mkdirs_if_not_exists, temp_chdir
from discoutils.thesaurus_loader import DenseVectors

if sys.version_info.major == 3:
    import iterpipes3 as iterpipes
else:
    import iterpipes


def set_stage_in_byblo_conf_file(filename, stage_id):
    """
    Add/removes the --stages switch from a Byblo conf filename
    :param stage_id: 0 if the --stages information should be removed, 1 if it has to be set to the first stage of
     Byblo (vector creation) and 2 for the second stage (all-pairs similarity)
    """
    with open(filename) as inf:
        lines = [x.strip() for x in inf.readlines()]
    stages = {
        0: '',  # run the entire Byblo pipeline
        1: ['--stages', 'enumerate,count,filter'],  # run the first part only
        2: ['--stages', 'allpairs,knn,unenumerate']  # run the second part only
    }

    # remove the current stages setting, may be multiple
    while True:
        try:
            index = lines.index('--stages')
            lines.pop(index)
            lines.pop(index)
        except ValueError:
            # '--stages' is not in list, nothing more to do
            break

    with open(filename, "w") as outf:
        for line in lines:
            outf.write(line)
            outf.write('\n')
        for line in stages[stage_id]:
            outf.write(line)
            outf.write('\n')


def set_output_in_byblo_conf_file(filename, new_output_prefix, type='output'):
    with open(filename) as inf:
        lines = [x.strip() for x in inf.readlines()]

    try:
        index = lines.index('--%s' % type)
    except ValueError:
        # not there, try the short form
        try:
            index = lines.index('-%s' % type[0])
        except ValueError:
            raise ValueError('Cannot find the "%s" parameter to Byblo in file %s' % (type, filename))
    lines.pop(index)
    lines.pop(index)

    with open(filename, "w") as outf:
        outf.write('--%s\n' % type)
        outf.write('%s\n' % new_output_prefix)

        for line in lines:
            outf.write(line)
            outf.write('\n')


def parse_byblo_conf_file(path):
    """
    Parses a byblo conf file (switch per line) and extracts a few important switches. Has only been configured
    to recognise

    :returns: a tuple of (known, unknown) arguments
    """
    with open(path) as infile:
        lines = ' '.join([x.strip() for x in infile.readlines()])
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", '--output', type=str)
    parser.add_argument("-i", '--input', type=str)
    parser.add_argument("-fef", '--filter-entry-freq', type=int)
    parser.add_argument("-fff", '--filter-feature-freq', type=int)
    parser.add_argument('--stages', type=str)
    args = parser.parse_known_args(lines.split(' '))
    return args


def get_byblo_out_prefix(conf_file):
    opts, _ = parse_byblo_conf_file(conf_file)
    return os.path.join(opts.output, os.path.basename(opts.input))


def touch_byblo_input_file(conf_file):
    opts, _ = parse_byblo_conf_file(conf_file)
    if not os.path.exists(opts.input):
        with open(opts.input, 'w') as inf:
            pass


def run_and_log_output(cmd_string, *args, **kwargs):
    """
    Runs a command with iterpipes and logs the output
    """
    c = iterpipes.cmd(cmd_string, bufsize=128, *args, **kwargs)
    logging.info('Running %s', iterpipes.format(cmd_string, args))
    out = iterpipes.run(c)
    for line in out:
        logging.info(line)


def run_byblo(conf_file, touch_input_file=False):
    """
    Runs Byblo with the specified configuration file
    :param conf_file: the Byblo conf file
    :param touch_input_file: if true, the input file specified in the conf file
    will be created. This is useful when only running the latter 3 stages of Byblo,
    that do no actually need an input file containing features (but need an events,
    entries and features files). Byblo still checks if the input file exists and
    complains
    """
    if touch_input_file:
        touch_byblo_input_file(conf_file)
    run_and_log_output('./byblo.sh @{}'.format(conf_file))


def unindex_all_byblo_vectors(outfile_name):
    """
    unindexes byblo's vector files to a string representation

    :param outfile_name: the name of the output file used when these vector files were produced
    """
    run_and_log_output('./tools.sh unindex-events -i {0}.events.filtered -o {0}.events.filtered.strings '
                       '-Xe {0}.entry-index -Xf {0}.feature-index -et JDBM'.format(outfile_name))
    run_and_log_output('./tools.sh unindex-features -et JDBM  -i {0}.features.filtered  '
                       '-o {0}.features.filtered.strings  -Xf {0}.feature-index -Ef'.format(outfile_name))
    run_and_log_output('./tools.sh unindex-entries -et JDBM  -i {0}.entries.filtered  '
                       '-o {0}.entries.filtered.strings  -Xe {0}.entry-index -Ee'.format(outfile_name))

    # remove the __FILTERED__ feature, entry and event so that it doesn't mess with cosine similarity
    for file_type in ['entries', 'features']:
        my_file = '{}.{}.filtered.strings'.format(outfile_name, file_type)
        with open(my_file) as infile:
            lines = infile.readlines()

        with open(my_file, 'w') as outfile:
            for line in lines:
                if '__FILTERED__' not in line:
                    outfile.write(line)
                else:
                    logging.info('Removed line %s from %s', line.strip(), my_file)

    events_file = '{}.events.filtered.strings'.format(outfile_name)
    with open(events_file) as infile:
        lines = infile.readlines()

    with open(events_file, 'w') as outfile:
        for line in lines:
            if not line.startswith('___FILTERED___'):
                outfile.write('\t'.join(line.split('\t')[:-2]))
                outfile.write('\n')
            else:
                logging.info('Removed line %s from %s', line.strip(), events_file)


def reindex_all_byblo_vectors(output_prefix):
    """rebuild index from a string representation"""
    run_and_log_output('./tools.sh index-features -et JDBM  -i {0}.features.filtered.strings  '
                       '-o {0}.features.filtered -Xf {0}.feature-index'.format(output_prefix))
    run_and_log_output('./tools.sh index-entries -et JDBM  -i {0}.entries.filtered.strings '
                       '-o {0}.entries.filtered -Xe {0}.entry-index'.format(output_prefix))
    run_and_log_output('./tools.sh index-events -et JDBM -i {0}.events.filtered.strings '
                       '-o {0}.events.filtered -Xe {0}.entry-index -Xf {0}.feature-index'.format(output_prefix))


def build_thesaurus_out_of_vectors(vectors_path, out_dir, threads=4, num_neighbours=100, sim_function='Cosine'):
    """
    Builds a Byblo thesaurus out of the provided vectors, however these were constructed. This function will make an
    uncompressed copy of the provided vectors file- might be slow and use up a lot of extra space.

    :param vectors_path: input vectors in byblo format, compressed or not
    :param out_dir: where to put the thesaurus and all temp file
    :param threads: number of byblo threads
    :param num_neighbours: number of nearest neighbours per entry to output
    :param sim_function: similarity measure between vectors to use. see byblo docs
    """
    from discoutils.thesaurus_loader import Vectors

    BYBLO_BASE_DIR = '/mnt/lustre/scratch/inf/mmb28/FeatureExtractionToolkit/Byblo-2.2.0'
    vectors_path = os.path.abspath(vectors_path)
    out_dir = os.path.abspath(out_dir)
    mkdirs_if_not_exists(out_dir)
    v = Vectors.from_tsv(vectors_path)

    # prepare the files that byblo expects
    outf_basename = os.path.join(out_dir, 'input')
    events_file = os.path.join(out_dir, outf_basename + '.events.filtered.strings')
    entries_file = os.path.join(out_dir, outf_basename + '.entries.filtered.strings')
    features_file = os.path.join(out_dir, outf_basename + '.features.filtered.strings')

    if isinstance(v, DenseVectors):
        # oh what a hack: DenseVectors do not natively support writing to plaintext (that byblo likes)
        # so let's pretend it's a Vectors object (replacing the self parameter)
        Vectors.to_tsv(v, events_file, entries_file, features_file, gzipped=False, dense_hd5=False)
    else:
        v.to_tsv(events_file, entries_file, features_file, gzipped=False)

    # write the byblo conf file
    conf = '--input {} --output {} --threads {} --similarity-min 0.01 -k {} ' \
           '--measure {} --stages allpairs,knn,unenumerate'.format(outf_basename, out_dir, threads,
                                                                   num_neighbours, sim_function)
    conf_path = os.path.join(out_dir, 'conf.txt')
    with open(conf_path, 'w') as outf:
        for line in conf.split():
            outf.write(line)
            outf.write('\n')

    # go baby go
    with temp_chdir(BYBLO_BASE_DIR):
        reindex_all_byblo_vectors(outf_basename)
        run_byblo(conf_path, touch_input_file=True)
        unindex_all_byblo_vectors(outf_basename)


def get_git_hash():
    c = iterpipes.cmd('git rev-parse HEAD')
    out = iterpipes.run(c)
    return list(out)[0].strip()