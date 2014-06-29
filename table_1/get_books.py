#!/usr/bin/env python
"""Download the top 100 project Gutenberg books

Create the `gut` directory and save the file
http://www.gutenberg.org/browse/scores/top in it.
"""

import re
import urllib


f = open('gut/top.htm')
top = f.read()
f.close()
ss = re.findall('"/ebooks/[0-9]+"', top)
id_books = [re.search('[0-9]+', s).group() for s in ss]

url_template = 'http://www.gutenberg.org/ebooks/%s.txt.utf8'
file_name_template = 'gut/%s.txt'

for id_book in id_books:
    url = url_template % id_book
    file_name = file_name_template % id_book
    print 'Dowloading', url
    urllib.urlretrieve(url, file_name)
print 'Done!'
