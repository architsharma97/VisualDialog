#!/bin/bash
# uncomment files to download: defaults to glove embeddings
# pre-trained glove embeddings with 400k words
# defaults to using the 300 dimensional embeddings from {50,100,200,300}
# change appropriately
# uncomment to delete the rest of the embeddings
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
mv glove.6B.300d.txt ../
rm -rf glove.6B*
mv ../glove.6B.300d.txt .

# pre-trained glove embeddings with 1.9M words and 300d embeddings
# wget http://nlp.stanford.edu/data/glove.42B.300d.zip

# pre-trained word2vec embeddings for 3M words and 300d embeddings
# follow the link: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
# place the downloaded bin.gz file in this folder (do not uncompress)