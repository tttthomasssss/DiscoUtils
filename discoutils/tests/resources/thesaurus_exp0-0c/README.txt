This is a Byblo 2.2.0 thesaurus built with ../exp0-0c.strings as the original events files.

The Python code to build it is (run in project root):

from glob import glob
import os
from discoutils.thesaurus_loader import Thesaurus, Vectors
from discoutils.collections_utils import walk_nonoverlapping_pairs
from discoutils.misc import temp_chdir
from thesisgenerator.scripts.build_phrasal_thesauri_offline import do_second_part_without_base_thesaurus

vectors_c = Vectors.from_tsv('discoutils/tests/resources/exp0-0c.strings')
events = os.path.abspath('tmp/test.events.filtered.strings')
entries = os.path.abspath('tmp/test.entries.filtered.strings')
features = os.path.abspath('tmp/test.features.filtered.strings')
vectors_c.to_tsv(events, entries, features)
thes_dir = os.path.abspath('tmp/thesaurus/')
conf_file = os.path.abspath('tmp/test.byblo.conf')
with temp_chdir('/Volumes/LocalDataHD/mmb28/NetBeansProjects/FeatureExtractionToolkit/Byblo-2.2.0'):
    do_second_part_without_base_thesaurus(conf_file, thes_dir,
                                          events, entries, features)
                                          
                                          
                                          
                                          
                                          
                                          
                                          
Then move the new directory tmp/thesaurus to discoutils/tests/resources and rename it to thesaurus_exp0-0c