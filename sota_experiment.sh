#!/bin/bash

"""
This script runs the best models from the paper accepted in
the 2017 WASSA workshop: Assessing State-of-the-art Sentiment Models on 
State-of-the-art Sentiment Datasets. Jeremy Barnes, Roman Klinger,
and Sabine Schulte im Walde.

The embeddings used are available at:
http://www.ims.uni-stuttgart.de/forschung/ressourcen/experiment-daten/sota-sentiment.html

If you use the code or resources, please cite the paper.
"""


# Setup Directories

if [[ ! -d results ]]; then
	mkdir results;
fi

if [[ ! -d predictions ]]; then
	mkdir predictions;
	for model in bow ave retrofit joint lstm bilstm cnn; do
		mkdir predictions/$model;
		for dataset in sst_fine sst_binary opener sentube_auto sentube_tablets semeval; do
			mkdir predictions/$model/$dataset;
			if [ "$model" = "lstm" ] || [ "$model" = "bilstm" ] || [ "$model" = "cnn" ]; then
				for i in 1 2 3 4 5; do
					mkdir predictions/$model/$dataset/run$i;
				done;
			fi
		done;
	done;
fi

if [[ ! -d embeddings ]]; then
	mkdir embeddings;
fi

# Get embeddings
cd embeddings

# Duyu Tang's Twitter-specific Sentiment Embeddings
# wget http://ir.hit.edu.cn/~dytang/paper/sswe/embedding-results.zip
# unzip embedding-results.zip
rm embedding-results.zip
rm embedding-results/sswe-h.txt
rm embedding-results/sswe-r.txt
mv embedding-results/sswe-u.txt .
rm -r embedding-results
cd ..


# Get Wikipedia, Google and Retrofit embeddings
# Currently, download them and add them manually



# Run Experiments

python3 bow.py -output results/results.txt
python3 ave.py -emb embeddings/google.txt -output results/results.txt
python3 retrofit.py -emb embeddings/retrofit.txt -output results/results.txt
python3 joint.py -emb emeddings/sswe-u.txt -file_type tang -output results/results.txt
