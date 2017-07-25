#!/bin/bash


"""
This script runs the best models from the paper accepted in
the 2017 WASSA workshop: Assessing State-of-the-art Sentiment Models on 
State-of-the-art Sentiment Datasets. Jeremy Barnes, Roman Klinger,
and Sabine Schulte im Walde.

The embeddings used are available at:
http://www.ims.uni-stuttgart.de/forschung/ressourcen/experiment-daten/sota-sentiment.html

If you use the code or resources, please cite the following paper:

bib:

@inproceedings{Barnes2017,
  author = {Barnes, Jeremy and Klinger, Roman and Schulte im Walde, Sabine},
  title = {Assessing State-of-the-Art Sentiment Models on State-of-the-Art Sentiment Datasets},
  booktitle = {Proceedings of the 8th Workshop on Computational
                  Approaches to Subjectivity, Sentiment and Social
                  Media Analysis},
  year = {2017},
  address = {Copenhagen, Denmark}
}
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


# Get Wikipedia embeddings
# Currently, download them and add them manually



# Run Experiments

python3 bow_log_reg.py -output results/results.txt