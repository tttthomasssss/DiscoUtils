# DiscoUtils
[![Build Status](https://travis-ci.org/mbatchkarov/DiscoUtils.svg?branch=master)](https://travis-ci.org/mbatchkarov/DiscoUtils)
[![Coverage Status](https://coveralls.io/repos/mbatchkarov/DiscoUtils/badge.svg?branch=master&service=github)](https://coveralls.io/github/mbatchkarov/DiscoUtils?branch=master)

A small collection of tools for DIStributional COmpositional semantics. I use this in my doctoral work, releasing it here in the hope that it might save somebody a bit of work. The API is inconsistent at times and the documentation may be outdated, but test coverage is decent and all tests pass.

# Use case 1: Wrangling word vectors

## Reading and storing dense and sparse word vectors
Let's read in a bunch of word vectors, stored in the sparse format used by [Byblo](https://github.com/MLCL/Byblo)


    !head -2 dud_vectors_sparse.txt

    council/N	pobj-HEAD:of/CONJ	31
    attack/N	pobj-HEAD:in/CONJ	23	pobj-HEAD:of/CONJ	58	amod-DEP:terrorist/J	21


We have one `entry` per row, with its `features` separated by a tab. In the example above, the entry `attack/N` was seen 21 times as the dependent of an `amod` of the word `terrorist/J`, etc. Let us read that in.


    from discoutils.thesaurus_loader import Vectors
    v = Vectors.from_tsv('dud_vectors_sparse.txt')
    v.get_vector('attack/N')




    <1x37 sparse matrix of type '<class 'numpy.float64'>'
    	with 3 stored elements in Compressed Sparse Row format>



The input file can also be gzipped or stored in an HDF file. The file type is determined automatically (the method is helpfully called `from_tsv` for historical reasons). High-dimensional vectors, such as the ones shown above, are best stored in gzipped sparse format. Low-dimensional dense vectors, such as those produced by `word2vec` or by applying SVD to the sparse vectors above are best stored in HDF format.

## Writing word vectors

Once we've read some word vectors, we can write them out in a range of formats, such as gzip, HDF, or [dissect](https://github.com/composes-toolkit/dissect)


    v.to_tsv('tmp.gz', gzipped=True);
    v.to_tsv('tmp.h5', dense_hd5=True);
    v.to_dissect_sparse_files('tmp');

## Efficient nearest neighbour search
We can measure the euclidean distance between any pair of entries:


    v.euclidean_distance('attack/N', 'council/N')




    41.2189276910499



We can also search for the nearest neighbours of an entry. This is implemented using a `BallTree` from `scikit-learn` for dense low-dimensional vectors and with brute-force matrix multiplication for high-dimensional ones. `BallTree` is significantly faster. At the time of writing (8 July 2015) sklearn's approximate nearest neighbour search is slower than `BallTree`. I have been meaning to experiment with [Annoy](https://github.com/spotify/annoy) but I haven't yet.


    v.init_sims()
    v.get_nearest_neighbours('attack/N')[:3]




    [('council/N', 41.218927691049899),
     ('people/N', 57.271284253105414),
     ('which/DET', 65.161338230579645)]



A slightly more realistic example


    v1 = Vectors.from_tsv('../../FeatureExtractionToolkit/word2vec_vectors/word2vec-wiki-15perc.unigr.strings.rep0')
    v1.init_sims()
    v1.get_nearest_neighbours('attack/N')[:3]




    [('raid/N', 1.3087977116653637),
     ('airstrike/N', 1.4726388902229308),
     ('assault/N', 1.6013899436574217)]



## Pointwise Mutual Information


    from discoutils.reweighting import ppmi_sparse_matrix
    ppmi_sparse_matrix(v.matrix)




    <13x37 sparse matrix of type '<class 'numpy.float64'>'
    	with 47 stored elements in Compressed Sparse Row format>



## Singular Value Decomposition


    from discoutils.reduce_dimensionality import do_svd
    do_svd('dud_vectors_sparse.txt', 'vectors_reduced', reduce_to=[5, 10], use_hdf=False)


    !head -1 vectors_reduced-SVD5.events.filtered.strings

    israel/N	SVD:feat001	4.21179787839	SVD:feat003	71.6348083843


# Running external processes
`DiscoUtils` has a bunch of utility function for running code in a separate process and capturing its output. The majority of these make it easy to run Byblo, but they are all built on top of the same building blocks:


    # reconfigure logging module
    import logging
    from discoutils.cmd_utils import run_and_log_output
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    run_and_log_output('head -2 dud_vectors_sparse.txt')

    INFO:root:Running head -2 dud_vectors_sparse.txt
    INFO:root:council/N	pobj-HEAD:of/CONJ	31
    attack/N	pobj-HEAD:in/CONJ	23	pobj-HEAD:of/CONJ	58	amod-DEP:terrorist/J	21
    
More examples coming soon.

# Other bits and pieces
Obscure and/or poorly documented features of `DiscoUtils`:
 - run Stanford [CoreNLP](http://nlp.stanford.edu/software/corenlp.shtml) on your data in parallel
 - find counting vectors for noun phrases from a corpus (similar to the first example above, but the entries are noun phrases instead of single words)
 

