#!/usr/bin/env python

import os
import sys
import logging
import string
import argparse

import numpy as np

from gensim import corpora, models, similarities
from gensim.matutils import sparse2full

def make_tfidf(mm, id2word, file_name, reset=False):
    """Build a TFIDF model.

    If the file_name of the model exists, load it, build a new one and save
    it as file_name otherwise.

    Parameters
    ----------
    mm : Corpus in BOW format.
    id2word : id2word corresponding associated to the corpus.
    file_name : File name of the model.
    reset : If True, make a new model regardles if file_name already exists.

    Return
    ------
    lsi : An TFIDF model.
    """
    if os.path.isfile(file_name) and not reset:
        tfidf = models.TfidfModel.load(file_name)
    else:
        tfidf = models.TfidfModel(mm, id2word=id2word, normalize=True)
        tfidf.save(file_name)
    return tfidf

def make_lsi(mm, id2word, file_name, reset=False):
    """Build an LSI model.

    If the file_name of the model exists, load it, build a new one and save
    it as file_name otherwise.

    Parameters
    ----------
    mm : Corpus in TFIDF format.
    id2word : id2word corresponding associated to the corpus.
    file_name : File name of the model.
    reset : If True, make a new model regardles if file_name already exists.

    Return
    ------
    lsi : An LSI model.
    """

    if os.path.isfile(file_name) and not reset:
        lsi = models.LsiModel.load(file_name)
    else:
        lsi = models.lsimodel.LsiModel(corpus=mm,
                                       id2word=id2word, num_topics=400)
        lsi.save(file_name)
    return lsi

def make_lda(mm, id2word, file_name, reset=False, online=True):
    """Build an LDA model.

    If the file_name of the model exists, load it, build a new one and save
    it as file_name otherwise.

    Parameters
    ----------
    mm : Corpus in BOW format.
    id2word : id2word corresponding associated to the corpus.
    file_name : File name of the model.
    reset : If True, make a new model regardles if file_name already exists.
    online : If True, use the online verison, use the batch version otherwise.

    Return
    ------
    lda : An LDA model.
    """
    if os.path.isfile(file_name) and not reset:
        lda = models.LdaModel.load(file_name)
    else:
        if online:
            lda = models.ldamodel.LdaModel(corpus=mm,
                                           id2word=id2word,
                                           num_topics=100,
                                           update_every=1,
                                           chunksize=10000,
                                           passes=1)
        else:
            lda = models.ldamodel.LdaModel(corpus=mm,
                                           id2word=id2word,
                                           num_topics=100,
                                           update_every=0,
                                           passes=20)
        lda.save(file_name)
    return lda


def make_index(corpus, model, file_name, reset=False):
    """Make an index.

    If the file_name of the index exists, load it, create a new one and save
    it as file_name otherwise.

    Parameters
    ----------
    corpus : List of BOW documents.
    model : Vector space model used to project the doc2bow representation.
    file_name : File name of the model.
    reset : If True, make a new index regardles if file_name already exists.

    Return
    ------
    index : A gensim index in matrix similarity format.
    """
    if os.path.isfile(file_name) and not reset:
        index = similarities.MatrixSimilarity.load(file_name)
    else:
        index = similarities.MatrixSimilarity(model[corpus])
        index.save(file_name)
    return index

def compute_similarity(doc, dictionary, model, index, tfidf=None):
    """Compute the similarity between a document and an index.

    Parameters
    ----------
    doc : Tokenized version of the document to compare.
    dictionary : Dictionary with the doc2bow method.
    model : Vector space model used to project the doc2bow representation
    index : Index of documents to compare with.
    tfidf : If not None, tfidf model used to project into TFIDF space before
            projection onto the model space.

    Return
    ------
    sims : List with the similarities between doc and the documents in the
           index.
    """
    vec_bow = dictionary.doc2bow(doc)
    # convert the query to model space
    if tfidf:
        vec_model = model[tfidf[vec_bow]]
    else:
        vec_model = model[vec_bow]
    sims = index[vec_model]
    return sims

def compute_similarity_hellinger(doc_1, doc_2, dictionary, model):
    """Compute the similarity between two documents using Hellinger similarity.

    Hellinger similarity between two vectors is computed as
    s = ||sqrt(p) - sqrt(q)|| / sqrt(2).

    Parameters
    ----------
    doc_1, doc_2 : Tokenized version of the documents to compare.
    dictionary : Dictionary with the doc2bow method.
    model : Vector space model used to project the do2bow representation.

    Return
    ------
    sim : Similarity between doc_1 and doc_2.
    """
    doc_1_bow = dictionary.doc2bow(doc_1)
    doc_2_bow = dictionary.doc2bow(doc_2)
    p = sparse2full(model[doc_1_bow], model.num_topics)
    q = sparse2full(model[doc_2_bow], model.num_topics)

    sim = np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)
    return sim

def check_files(files):
    """Check if the files exist.

    If one or more file in files does not exist, exit.

    Parameters
    ---------
    files : List of files to check.
    """
    for f in files.values():
        if not os.path.isfile(f):
            print 'File %s does not exist.' % f
            print 'You probably need to run '
            print ('$ ./wikicorpus.py ?????-?????-pages-articles.xml.bz2 '
                   'OUTPUT_PREFIX')
            sys.exit(1)

def config_logger(log_name):
    """Config the logger to log both to a file and the console.

    Parameters
    ----------
    log_name : Name of the log file.
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_name)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate LSA/LDA model')
    parser.add_argument('-test', action='store_true', default=False,
                        help='Test the model.')
    parser.add_argument('-model', default='LSA',
                        choices=['LSA', 'LDA_ONLINE', 'LDA_BATCH'],
                        help='Model to be used. Default to LSA')
    parser.add_argument('-prefix', default='results/simplewiki',
                        help='Prefix used with the file names. Default to '
                        'results/simplewiki.')
    parser.add_argument('-log_name', default='model.log',
                        help='Name of the log file')
    parser.add_argument('-hellinger', action='store_true', default=False,
                        help='Use Hellinger similarity')
    args = parser.parse_args()

    logger = config_logger(args.log_name)
    prefix = args.prefix

    if args.hellinger and args.model == 'LSA':
        print 'Hellinger similarity only works with LDA models for the moment.'
        sys.exit(0)

    # Set the file names
    files = {
        'wordids' : prefix + '_wordids.txt.bz2',
        'mm_tfidf' : prefix + '_tfidf.mm',
        'mm_bow' : prefix + '_bow.mm',
        }

    new_files = {
        'lsi' : prefix + '_model.lsi',
        'lda_online': prefix + '_model.ldao',
        'lda_batch': prefix + '_model.ldab',
        'tfidf' : prefix + '_model.tfidf',
        'index' : prefix + '.index',
        }

    check_files(files)

    id2word = corpora.Dictionary.load_from_text(files['wordids'])
    mm_tfidf = corpora.MmCorpus(files['mm_tfidf'])
    mm_bow = corpora.MmCorpus(files['mm_bow'])
    if args.model == 'LSA':
        model = make_lsi(mm_tfidf, id2word, new_files['lsi'])
    elif args.model == 'LDA_ONLINE':
        model = make_lda(mm_bow, id2word, new_files['lda_online'])
    elif args.model == 'LDA_BATCH':
        model = make_lda(mm_bow, id2word, new_files['lda_batch'],
                         online=False)

    # We need to recreate the tfidf model
    # Shouldn't wikicorpus save the model when it builds it?

    tfidf = make_tfidf(mm_bow, id2word, new_files['tfidf'])

    # Test the model by computing the similarity between the 'army' article in
    # the simple wikipedia, and three other articles
    if args.test:
        text = corpora.wikicorpus.filter_wiki(open('army.wiki').read())
        text = corpora.wikicorpus.tokenize(text)
        # Index with one document
        if args.model == 'LSA':
            corpus_ind = [tfidf[id2word.doc2bow(text)]]
        else:
            corpus_ind = [id2word.doc2bow(text)]
        index = make_index(corpus_ind, model, new_files['index'], reset=False)

        from docs import docs
        for doc, desc in docs:
            doc = corpora.wikicorpus.filter_wiki(doc)
            doc = corpora.wikicorpus.tokenize(doc)
            if args.model == 'LSA':
                sim = compute_similarity(doc, id2word, model, index, tfidf)
            else:
                if args.hellinger:
                    sim = compute_similarity_hellinger(doc, text, id2word,
                                                       model),
                else:
                    sim = compute_similarity(doc, id2word, model, index)
            print '%s -> %f' % (desc, sim[0])
