__author__ = 'thk22'
import numpy as np


def _create_template(b):
	# Possibly involves some overhead but its fun and I don't want to clutter this beautiful function with if statements
	s = np.array(['{}', '\t{}', '\\{}', '\\{}', '\t{}', '\t{}'])

	return ''.join(s[b])


def convert_stanford_conll_to_apt_conll(in_path, out_path, lowercase=True, include_pos_tags=False, include_ner_tags=False):
	# Lol! \m/o_O\m/
	b_template = np.array([True, True, include_pos_tags, include_ner_tags, True, True])
	b_context = np.array([True, True, False, include_pos_tags, include_ner_tags, True, True])
	template = _create_template(b_template)

	with open(in_path, 'rb') as in_file, open(out_path, 'wb') as out_file:
		for line in in_file:
			if (line.strip() != ''):
				p = np.array(map(lambda x: x.lower().replace('_', ''), line.split('\t')))
				out_line = template.format(*p[b_context])
				out_file.write(out_line)
			else:
				out_file.write(line)

if (__name__ == '__main__'):
	in_path = '/Volumes/LocalDataHD/thk22/DevSandbox/InfiniteSandbox/_datasets/cat_test-tagged/cat/testsentence.tagged'
	out_path = '/Volumes/LocalDataHD/thk22/DevSandbox/InfiniteSandbox/_datasets/cat_test-tagged/cat/testsentence.apt.conll.txt'
	convert_stanford_conll_to_apt_conll(in_path, out_path, None)