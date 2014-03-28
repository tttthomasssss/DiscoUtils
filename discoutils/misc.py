__author__ = 'miroslavbatchkarov'


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