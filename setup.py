# -*- coding: utf-8 -*-
from setuptools import setup
from setuptools.command.test import test as TestCommand
from Cython.Build import cythonize
import pytest


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = 'discoutils/tests'
        self.test_suite = True

    def run_tests(self):
        pytest.main(self.test_args)

setup(
    name='DiscoUtils',
    version='0.3',
    packages=['discoutils', 'discoutils.tests'],
    author=['Julie Weeds', 'Miroslav Batchkarov'],
    author_email=['J.E.Weeds@sussex.ac.uk', 'M.Batchkarov@sussex.ac.uk'],
    tests_require=['pytest>=2.4.2'],
    cmdclass={'test': PyTest},
    install_requires=['pytest', 'Cython', 'iterpipes3', 'numpy', 'scipy',
                      'scikit-learn', 'six', 'python-magic'],
    ext_modules=cythonize(["discoutils/tokens.pyx"])
)

