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

input=""
case ${data} in 
    rcv-wiki) input="/iesl/canvas/jshankar/rcv-wiki_clean.txt";;
    rcv-wiki-phrase) input="/iesl/canvas/jshankar/socher-data/rcv-wiki-phrase-3";;
    socher-wiki-match) input="/iesl/canvas/jshankar/socher-data/socher-wiki-match.txt";;
    text8) input=text8_linebreak ;; 
esac
bootvocab="/iesl/canvas/jshankar/vectors/vocab/${data}.vocab.gz"
multivocab="/iesl/canvas/jshankar/vectors/vocab/socher-only-multi.vocab"
output="vec_${data}_${dim}d_${num_sense}s_${num_neg}num_neg_${v}v_${sam}sam_${stopwords}stopwords_${mv}mv.txt.gz"


${wordvec_app} --use-k-means=1 --learn-top-v=${mv} --load-multi-vocab=${multivocab} --cbow=2 --rate=0.025 --train ${input} --output ${output} --size=${dim} --window=5  --min-count=11 --threads=23 --negative=${num_neg} --sample=${sam} --sense=${num_sense} --load-vocab=${bootvocab} --binary=1 --ignore-stopwords=${stopwords} --encoding=ISO-8859-15 --max-vocab-size=$v
