__author__ = 'miroslavbatchkarov'

import os, contextlib

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
    Delays a function call
    >>> d = Delayed(int, '123')
    >>> # do some more things here
    >>> d()
    123
    """

    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.fn(*self.args, **self.kwargs)

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