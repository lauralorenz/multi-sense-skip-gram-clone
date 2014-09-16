#!/bin/sh 

./make_cp.sh   ## compile and make class-path
./train-mssg.sh   ## train the embeddings 
./browse-mssg.sh vectors_MSSG.gz 2 20 ## browse the nearest-neighbours
