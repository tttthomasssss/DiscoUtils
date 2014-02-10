__author__ = 'miroslavbatchkarov'


class ContainsEverything(object):
    """
    A drop-in replacement for a set that thinks it contains everything
    """
    def __contains__(self, item):
        return True