import magic

__author__ = 'miroslavbatchkarov'

import os
import contextlib


class Bunch:
    """
    "collector of a bunch of named stuff" class
    http://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/?in=user-97991
    """

    def __init__(self, **kwds):
        self.__dict__.update(kwds)


class ContainsEverything(object):
    """
    A drop-in replacement for a set that thinks it contains everything
    """

    def __contains__(self, item):
        return True


class Delayed(object):
    """
    Delays a function call. Passing in the object that the function is bound to makes this object picklable.
    Modified from http://stackoverflow.com/a/1816969/419338
    >>> d = Delayed(int, '123')
    >>> # do some more things here
    >>> d()
    123
    """

    def __init__(self, obj, method, *args, **kwargs):
        self.obj = obj
        self.args = args
        self.kwargs = kwargs
        if isinstance(method, str):
            self.methodName = method
        else:
            assert callable(method)
            self.methodName = method.__name__ # was called func_name in python2

    def __call__(self, *args, **kwargs):
        return getattr(self.obj, self.methodName)(*self.args, **self.kwargs)


@contextlib.contextmanager
def temp_chdir(path):
    """
    Source: http://orip.org/2012/07/python-change-and-restore-working.html
    Usage:
    >>> with temp_chdir(gitrepo_path):
    ...   subprocess.call('git status')
    """
    starting_directory = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(starting_directory)

def mkdirs_if_not_exists(dir):
    """
    Creates a directory (and all intermediate directories) if it doesn't exists.
    Behaves like mkdir -p, and is prone to race conditions

    Source: http://stackoverflow.com/q/273192/419338
    :param dir:
    :return:
    """
    if not (os.path.exists(dir) and os.path.isdir(dir)):
        os.makedirs(dir)

def is_gzipped(path_to_file):
    """
    Checks if a file is gzipped by looking at its magic number (requires libmagic). Follows symlinks
    :param path_to_file: may be a symlink
    """
    return b'gzip compressed data' in magic.from_file(os.path.realpath(path_to_file))