from itertools import tee


def walk_overlapping_pairs(iterable):
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ...

    From http://docs.python.org/2/library/itertools.html
    """

    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def walk_nonoverlapping_pairs(iterable, beg, max_pairs=1e20):
    """
    s -> (s0,s1), (s2,s3), ..., yielding at most max_pair tuples
    """
    for tuple_number, index in enumerate(range(beg, min(len(iterable) - 1, len(iterable)), 2)):
        if tuple_number <= max_pairs - 1:
            yield (iterable[index], iterable[index + 1])

