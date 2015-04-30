from discoutils.misc import is_gzipped, is_hdf


def test_is_gzipped():
    assert not is_gzipped('discoutils/tests/resources/exp0-0a.strings')
    assert is_gzipped('discoutils/tests/resources/exp0-0a.strings.gzip')


def test_is_hdf(tmpdir):
    import pandas as pd

    assert not is_hdf('discoutils/tests/resources/exp0-0a.strings')
    assert not is_hdf('discoutils/tests/resources/exp0-0a.strings.gzip')

    tmpfile = tmpdir.join('tmp')
    df = pd.DataFrame(dict(a=[1, 2, 3], b=[1, 2, 3]))
    with pd.get_store(tmpfile.strpath, mode='w',
                      complevel=9, complib='zlib') as store:
        store['matrix'] = df

    assert is_hdf(tmpfile.strpath)
