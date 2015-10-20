# coding=utf-8
import argparse
import logging
import os
import subprocess
import datetime as dt
from discoutils.misc import mkdirs_if_not_exists


def current_time():  # for reporting purposes.
    return dt.datetime.ctime(dt.datetime.now())


def _make_filelist_and_create_files(data_dir, filelistpath, output_dir):
    """
    1. Create a list of files in a directory to be processed, which
       can be passed to stanford's "filelist" input argument.
    2. Pre-create each output file in an attempt to avoid cluster
       problems.
    """
    with open(filelistpath, 'w') as filelist:
        for filename in os.listdir(data_dir):
            if not filename.startswith("."):
                filepath = os.path.join(data_dir, filename)
                filelist.write("%s\n" % filepath)
                with open(os.path.join(output_dir, filename + ".tagged"),
                          'w'):
                    pass


def run_stanford_pipeline(data_dir, stanford_dir, java_threads=2,
                          filelistdir=""):
    """
    Process directory of text using stanford core nlp
    suite. Perform:
        - Tokenisation
        - Sentence segmentation
        - PoS tagging
        - Lemmatisation

    Output CONLL to "*data_dir*-tagged"
    """
    if not all([data_dir, stanford_dir]):
        raise ValueError("ERROR: Must specify path to data and stanford tools.")

    # Create output directory
    output_dir = "%s-tagged" % data_dir
    try:
        os.mkdir(output_dir)
    except OSError:
        pass  # Directory already exists

    # Change working directory to stanford tools
    os.chdir(stanford_dir)

    logging.info("<%s> Beginning stanford pipeline..." % current_time())

    for data_sub_dir in [name for name in os.listdir(data_dir) if
                         not name.startswith(".")]:
        # Setup output subdirectory
        output_sub_dir = os.path.join(output_dir, data_sub_dir)
        input_sub_dir = os.path.join(data_dir, data_sub_dir)
        mkdirs_if_not_exists(output_sub_dir)

        # Create list of files to be processed.
        filelist = os.path.join(filelistdir if filelistdir else stanford_dir,
                                "%s-filelist.txt" % data_sub_dir)
        _make_filelist_and_create_files(input_sub_dir, filelist, output_sub_dir)

        logging.info("<%s> Beginning stanford processing: %s" % (
            current_time(), input_sub_dir))

        # Construct stanford java command.
        stanford_cmd = ['./corenlp.sh', '-annotators',
                        'tokenize,ssplit,pos,lemma,parse',
                        '-filelist', filelist,
                        '-outputDirectory', output_sub_dir,
                        '-threads', str(java_threads), '-outputFormat', 'conll',
                        '-outputExtension', '.tagged', '-parse.maxlen', '50']

        logging.info("Running: \n" + str(stanford_cmd))

        # Run stanford script, block until complete.
        subprocess.call(stanford_cmd)
        logging.info("<%s> Stanford complete for path: %s" % (current_time(), output_sub_dir))

    logging.info("<%s> All stanford complete." % current_time())
    return output_dir


def execute_pipeline(path_to_corpora, path_to_stanford,
                     path_to_filelistdir="", java_threads=40):
    if not path_to_stanford:
        raise ValueError("Specify path to stanford")
    run_stanford_pipeline(path_to_corpora, path_to_stanford,
                          java_threads, path_to_filelistdir)

    logging.info("Pipeline finished")


if __name__ == "__main__":
    """
    ---- Resources Required ----

    This section lists software required to
    run annotate_corpora.py.

        2. Python 3
        4. Stanford CoreNLP pipeline
        5. Python's joblib package

        - How to acquire resources:
            1. Stanford CoreNLP pipeline
                - Download from: http://nlp.stanford.edu/software/corenlp.shtml

            2. Joblib
                - Download from: http://pypi.python.org/pypi/joblib or
                install with pip


    ---- Execution ----
    This section explains how to run stanford_utils.py

        - Expected Input
            The pipeline expects input data in the following structure:
                - A directory containing corpora, where
                - Each corpus is a directory of files, where
                - Each file contains raw text.
        - Output

            After running the full pipeline on a directory called "corpora"
            You should see the following output:

                - A directory called "corpora-tagged" contains a version of
                  your data in CoNLL style format after the execution of the
                  following parts of stanford corenlp:

                    - Tokenization
                    - Sentence segmenation
                    - Lemmatisation
                    - PoS tagging

                - A directory called "corpora-tagged-parsed" which adds the
                  annotations of AR's dependency parser to the data.

        - Invocation using "execute_pipeline" function. The the following arguments are required:

            - path_to_stanford
                This is the full path to the directory containing Stanford CoreNLP

            - stanford_java_threads
                The number of threads to be used when running stanford corenlp

                DEFAULT: 2
    """

    # Pipeline examples:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s")

    parser = argparse.ArgumentParser(description='Process labelled data set with Stanford CoreNLP toolkit')
    parser.add_argument('--data', help='Path to labelled data directory')
    parser.add_argument('--stanford', help='Path to Stanford toolkit. Must include corenlp.sh')
    parser.add_argument('--threads', help='Path to Stanford toolkit. Must include corenlp.sh',
                        type=int, default=2)

    args = parser.parse_args()
    execute_pipeline(os.path.abspath(args.data), os.path.abspath(args.stanford), java_threads=args.threads)
