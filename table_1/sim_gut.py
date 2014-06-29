#!/usr/bin/env python

# Compute semantic similarity using the top 100 Gutember books

import string
import logging
import os
from pprint import pprint

from gensim import corpora, models, similarities

# Log both to the file and the console
log_name = 'sim_gut.log'
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

# base_dir must be a directory with the top 100 most popular Gutenberg proyect
# books in txt format. The books can be downloaded using the script get_books.py
base_dir = '/home/ajw/Dropbox/projects/inactive/active_cs/python/logical_depth/gut/'
file_list = [f for f in os.listdir(base_dir) if f.count('txt')]

def get_text():
    for file_name in file_list:
        f = open(os.path.join(base_dir, file_name))
        s = f.read().decode('utf-8').encode('ascii', 'ignore')
        f.close()
        yield s.translate(string.maketrans("",""), string.punctuation)

def make_corpus_dict(file_name='gut.dict', reset=False):
    if os.path.isfile(file_name) and not reset:
        dictionary = corpora.Dictionary.load(file_name)
        print 'Loaded %s dictionary.' % file_name
    else:
        dictionary = corpora.Dictionary(s.lower().split() for s in get_text())
        # remove stop words and words that appear only once
        #stoplist = set('for a of the and to in or'.split())
        stoplist = set([w.strip().lower() for w in open('common_words.txt')])
        stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
                    if stopword in dictionary.token2id]
        once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems()
                    if  docfreq == 1]
        dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and
                                                     # words that appear only
                                                     # once
        dictionary.compactify() # remove gaps in id sequence after words that
                                # were removed
        dictionary.save('gut.dict')
    return dictionary

def make_corpus(dictionary, file_name='gut.mm', reset=False):
    if os.path.isfile(file_name) and not reset:
        corpus = corpora.MmCorpus(file_name)
    else:
        corpus = [dictionary.doc2bow(text.lower().split()) for text in
                  get_text()]
        corpora.MmCorpus.serialize(file_name, corpus)
    return corpus

def make_lsi(dictionary, corpus, file_name='gut.lsi', reset=False):
    if os.path.isfile(file_name) and not reset:
        lsi = models.LsiModel.load(file_name)
    else:
        lsi = models.LsiModel(corpus, id2word=dictionary)#, numTopics=200)
        lsi.save(file_name)
    return lsi

def make_index(corpus, lsi, file_name='gut.index', reset=False):
    if os.path.isfile(file_name) and not reset:
        index = similarities.MatrixSimilarity.load(file_name)
    else:
        index = similarities.MatrixSimilarity(lsi[corpus])
        index.save(file_name)
    return index

def compute_similarity(doc, dictionary, lsi, index):
    doc = doc.translate(string.maketrans("",""), string.punctuation)
    vec_bow = dictionary.doc2bow(doc.lower().split())
    vec_lsi = lsi[vec_bow] # convert the query to LSI space
    sims = index[vec_lsi] #####
    return sims

if __name__ == '__main__':
    dictionary = make_corpus_dict(reset=False)
    corpus = make_corpus(dictionary, reset=False)
    lsi = make_lsi(dictionary, corpus, reset=True)
    moby_dick_id = file_list.index('2701.txt')
    index = make_index([corpus[moby_dick_id]], lsi, reset=True)
    #index = make_index(corpus, lsi, reset= True)
    
    from docs import docs
    print
    for doc, desc in docs[1:]:
        sim = compute_similarity(doc, dictionary, lsi, index)
        print '%s -> %f' % (desc, sim[0])
        pass
    
