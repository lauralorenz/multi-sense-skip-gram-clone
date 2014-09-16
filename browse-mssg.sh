#!/bin/sh
## scipt for computing the vectors  and checking the accuracy with google's questions-words analogy task

if [ ! -e CP.hack ]; then
    echo "Run make_cp.sh script first"
    exit 
fi

classpath=`cat CP.hack`
nearest_neighbour_app="java -Xmx100g -cp ${classpath} MSEmbeddingDistance"

embedding_file=$1
number_of_senses=$2
number_of_nearest_neighbours=$3

${nearest_neighbour_app} ${embedding_file} ${number_of_senses} ${number_of_nearest_neighbours} 0 1
