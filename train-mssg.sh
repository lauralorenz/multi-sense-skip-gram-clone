#!/bin/sh

if [ ! -e text8 ]; then
  wget https://dl.dropboxusercontent.com/u/39534006/text8.zip
  unzip text8.zip
fi

if [ ! -e CP.hack ]; then
    echo "Run make_cp.sh script first"
    exit    
fi

classpath=`cat CP.hack`
wordvec_app="java -Xmx100g -cp ${classpath} WordVec"

${wordvec_app}  --train=text8 \                ## input corpus
		--output=vectors-MSSG.gz \     ## embedding output
                --model=MSSG-KMeans  \         ## MSSG model described in the paper. Try MSSG-MaxOut to do MSSG without cluster centers
		--sense=2 \                    ## number of senses 
		--learn-top-v=4000 \           ## learn multiple embeddings only for top-v words and single embeddings for remainining words in the vocab
		--size=300 \                   ## embedding dimentionality
		--window=5  \                  ## context winow length
		--min-count=11 \               ## ignores all the words whose frequeny is less 10
		--threads=23 \                 ## = number of cores in the machine
		--negative=1 \                 ## number of negative examples
		--sample=0.001 \               ## sub-sampling
		--binary=1 \                   ## stores the embedding and vocab in gzip
		--ignore-stopwords=1 \         ## ignores the stopwords
		--encoding=ISO-8859-15 \       ## encoding for reading the corpus
		--save-vocab=text8.vocab.gz \  ## next time , do --load-vocab=text8.vocab.gz 
                --rate=0.025 \                 ## ada-grad rate
                --delta=0.1                    ## ada-grad delta 
