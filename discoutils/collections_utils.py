from itertools import tee, izip


def walk_overlapping_pairs(iterable):
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ...

    From http://docs.python.org/2/library/itertools.html
    """

    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


def walk_nonoverlapping_pairs(iterable, beg):
    '''
    s -> (s0,s1), (s2,s3), ...
    '''
    for i in xrange(beg, min(len(iterable) - 1, len(iterable)), 2):  # step size 2
        yield (iterable[i], iterable[i + 1])

