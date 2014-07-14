# -*- coding: utf-8 -*-
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='DiscoUtils',
    version='0.23',
    packages=['discoutils', 'discoutils.tests'],
    author=['Julie Weeds', 'Miroslav Batchkarov'],
    author_email=['J.E.Weeds@sussex.ac.uk', 'M.Batchkarov@sussex.ac.uk'],
    install_requires=['pytest', 'Cython', 'iterpipes', 'numpy', 'scipy', 'scikit-learn'],
    ext_modules=cythonize(["discoutils/tokens.pyx"])
)

