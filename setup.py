# -*- coding: utf-8 -*-
from setuptools import setup, Command
from Cython.Build import cythonize

# https://pytest.org/latest/goodpractises.html#integrating-with-setuptools-python-setup-py-test
class PyTest(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        import sys
        errno = subprocess.call([sys.executable, 'runtests.py'])
        raise SystemExit(errno)


setup(
    name='DiscoUtils',
    version='0.3',
    packages=['discoutils', 'discoutils.tests'],
    author=['Julie Weeds', 'Miroslav Batchkarov'],
    author_email=['J.E.Weeds@sussex.ac.uk', 'M.Batchkarov@sussex.ac.uk'],
    tests_require=['pytest>=2.4.2'],
    cmdclass={'test': PyTest},
    install_requires=['pytest', 'Cython', 'iterpipes3', 'numpy', 'scipy',
                      'scikit-learn', 'joblib', 'python-magic', 'pandas'],
    ext_modules=cythonize(["discoutils/tokens.pyx"])
)
