#!/bin/sh
## scipt for computing the vectors  and checking the accuracy with google's questions-words analogy task

classpath=`cat CP.hack`
nearest_neighbour_app="java -Xmx100g -cp ${classpath} MSEmbeddingDistance"
${nearest_neighbour_app} $1 $2 20 0 $3
