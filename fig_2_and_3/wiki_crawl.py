#!/usr/bin/env python

import logging
import sys
import time
from operator import itemgetter

import mwclient
import networkx as nx

import vsm

only_ascii = lambda s: ''.join(i for i in s if ord(i)<128)

def secs2str(s):
    """Convert seconds s to string HH:MM:SS."""
    h = int(s/3600)
    s -= h*3600
    m = int(s/60)
    s = int(s - m*60)
    return ('%s:%s:%s' % (h, m, s))

class Crawler(object):
    def __init__(self, seed, site='en.wikipedia.org',
                 prefix='results/simplewiki'):
        self.site = mwclient.Site(site)
        self.page = self.site.Pages[seed]
        self.seed = seed
        logging.info('Crawling with seed %s', seed)
        self.G = nx.DiGraph()

        # Create an LSI instance to compute the similarities
        self.lsi = vsm.LSI(self.page.edit().encode('utf-8'), prefix)

    def crawl(self, n_max=1000):
        n = 0
        links = [(self.page, -1, None)]

        start = time.time()
        while n < n_max:
            link = links.pop()
            page = link[0]
            dist = link[1] + 1
            page_text = page.edit().encode('utf-8')
            v_lsi = self.lsi.get_lsi(page_text) # LSI representation of
                                                # page_text
            if len(v_lsi) == 0:
                continue
            self.G.add_node(only_ascii(page.name), v=v_lsi)
            if link[2]:
                source = only_ascii(link[2])
                dest = only_ascii(page.name)
                if self.G.has_edge(source, dest):
                    logging.info('Link %s -> %s already exists', source, dest)
                    continue
                else:
                    sim = self.lsi.get_similarity(page_text)[0]
                    if sim == 0: # To fix a bug in Networkx
                        sim = 0.01
                    self.G.add_edge(source,
                                    dest,
                                    weight=sim,
                                    d=dist)
                    logging.info('Adding link %s -> %s with similarity %.2f ' \
                                 'and distance %d. Link %d out of %d.',
                                 source, dest, sim, dist, n, n_max)
            else:
                logging.info('Adding node %s', only_ascii(page.name))
            # I filter out pages with a ':' in the name, to avoid pages like
            # 'Template:', 'Category:', etc. There must be a better way to do
            # this. It is possible that I am eliminating legitimate pages
            new_links = [(l, dist, page.name) for l in page.links() if
                         l.name.count(':') == 0]
            links = new_links + links
            n += 1
        dt = time.time() - start
        logging.info('Done crawling. Seed: %s, elapsed time: %s, n_crawl: %s' %
                     (self.seed, secs2str(dt), n_max))
        self.links = links

def filter_G(G):
    """Remove the nodes attributes of G."""
    G1 = nx.DiGraph()
    G1.add_nodes_from(G.nodes())
    G1.add_edges_from(G.edges(data=True))

    return G1

if __name__ == '__main__':
    # Log both to a file and the console
    log_name = 'wiki_crawl.log'
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

    #TODO: Use optoparse
    if len(sys.argv) != 5:
        print 'Usage: %s n_crawl seed wiki_site prefix' % sys.argv[0]
        sys.exit(0)

    n_crawl, seed, wiki_site, prefix = sys.argv[1:]

    crawler = Crawler(seed, wiki_site, prefix)
    crawler.crawl(int(n_crawl))

    graph_fname = 'graph.gexf'

    # TODO: Save the lsi vector optionally, and choose the format to save
    # accordingly.

    #nx.write_gexf(crawler.G, graph_fname)
    #logging.info('Saved graph as %s', graph_fname)
    #nx.write_graphml(crawler.G, 'graph.graphml')
    #nx.write_gml(crawler.G, 'graph.gml')
    nx.write_gpickle(crawler.G, 'graph.pickle')
    nx.write_gexf(filter_G(crawler.G), 'graph.gexf')
