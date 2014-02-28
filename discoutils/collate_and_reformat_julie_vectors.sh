#!/bin/bash
# converts Julie's observed vectors to my underscore_separated format
# absurdity/N:amod-DEP:total/J -> total/J_absurdity/N
# academy/N:nn-HEAD:award/N -> academy/N_award/N

cd /mnt/lustre/scratch/inf/mmb28/thesisgenerator

# put all files from the same source corpus together
x=/mnt/lustre/scratch/inf/juliewe/Compounds/data/miro/composed_vectors/
y=/mnt/lustre/scratch/inf/mmb28/FeatureExtractionToolkit/apdt_vectors/
cat $x/*wiki* > $y/exp11-12_AN_NNvectors
cat $x/*giga* > $y/exp10-12_AN_NNvectors

# convert to underscore-separated
python -c "
import re; 
from discoutils.io_utils import reformat_entries, clean;
for i in [10,11]:
    observed_ngram_vectors_file = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/apdt_vectors/exp%d-12_AN_NNvectors' % i;
    reformat_entries(observed_ngram_vectors_file, '-cleaned', clean);
"
