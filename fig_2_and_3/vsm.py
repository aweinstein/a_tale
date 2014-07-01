"""Class for computing the similarity between a document against a given
document previously indexed using Vector Space Modelling, such as Latent
Semantic Indexing (LSI) or Latent Dirchlet Allocation (LDA)."""

import os.path
import logging
import string

from gensim import corpora, models, similarities

class FileError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return 'Error reading %s' % self.value

class LSI(object):
    def __init__(self, corpus_ind, prefix):
        self.corpus_ind = corpus_ind
        fname = lambda sufix: prefix + sufix
        wordid_fn = fname('_wordids.txt')
        tfidf_fn = fname('_model.tfidf')
        lsi_fn = fname('_model.lsi')
        try:
            self.d = corpora.Dictionary.load_from_text(wordid_fn)
        except IOError as (errno, strerror):
            print "I/O error({0}): {1}".format(errno, strerror)
            raise FileError(wordid_fn)
        try:
            self.tfidf = models.TfidfModel.load(tfidf_fn)
        except IOError as (errno, strerror):
            print "I/O error({0}): {1}".format(errno, strerror)
            raise FileError(tfidf_fn)
        try:
            self.lsi = models.LsiModel.load(lsi_fn)
        except IOError as (errno, strerror):
            print "I/O error({0}): {1}".format(errno, strerror)
            raise FileError(lsi_fn)

        self.make_index(corpus_ind)

    def make_index(self, corpus):
        text_filter = corpora.wikicorpus.filter_wiki(corpus)
        text = corpora.wikicorpus.tokenize(text_filter)
        corpus = [self.tfidf[self.d.doc2bow(text)]]
        self.index = similarities.MatrixSimilarity(self.lsi[corpus])

    def get_similarity(self, text):
        vec_lsi = self.get_lsi(text)
        sims = self.index[vec_lsi]
        return sims

    def get_lsi(self, text):
        text = text.translate(string.maketrans("",""), string.punctuation)
        # convert the query to LSI space: bow -> tfidf -> lsi
        vec_bow = self.d.doc2bow(text.lower().split())
        vec_tfidf = self.tfidf[vec_bow]
        vec_lsi = self.lsi[vec_tfidf]
        return vec_lsi

if __name__ == '__main__':
    print 'Testing the LSI class'
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    corpus_ind = open('army.wiki').read()
    prefix = 'results/simplewiki'
    lsi = LSI(corpus_ind, prefix)

    # Compute the similarity between the army and all the documents in docs.py
    from docs import docs
    for doc, desc in docs:
        sim = lsi.get_similarity(doc)[0]
        print '%s -> %f' % (desc, sim)

    # We can also get the LSI representation of a text
    v_lsi = lsi.get_lsi(doc)
    v = sorted(v_lsi, key=lambda x: abs(x[1]), reverse=True)
