#!/bin/sh
## scipt for computing the vectors  and checking the accuracy with google's questions-words analogy task

if [ ! -e text8 ]; then
  wget https://dl.dropboxusercontent.com/u/39534006/text8.zip
  unzip text8.zip
fi

classpath=`cat CP.hack`
wordvec_app="java -Xmx100g -cp ${classpath} WordVec"


data="text8"
dim="300"
num_sense="2"
num_neg="1"
v="70000"
sam="0.001"
stopwords="1"
mv="5000"

#bootvocab="/iesl/canvas/jshankar/vectors/vocab/${data}.vocab.gz"
#multivocab="/iesl/canvas/jshankar/vectors/vocab/socher-only-multi.vocab"
#output="vec_${data}_${dim}d_${num_sense}s_${num_neg}num_neg_${v}v_${sam}sam_${stopwords}stopwords_${mv}mv.txt.gz"


${wordvec_app} --use-k-means=1 --learn-top-v=2000 --cbow=2 --train text8 --output vectors_MSSG.gz --size=300 --window=5  --min-count=11 --threads=23 --negative=1 --sample=0.001 --sense=2 --binary=1 --ignore-stopwords=1 --encoding=ISO-8859-15 --save-vocab=text8.vocab.gz
