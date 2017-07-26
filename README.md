# Assessing State-of-the-art Sentiment Models on State-of-the-art Sentiment Datasets

Jeremy Barnes [jeremy.barnes@upf.edu] / [barnesjy@ims.uni-stuttgart.de]

This experiment runs the best models with the best embeddings as described in the following paper:

Jeremy Barnes, Roman Klinger, and Sabine Schulte im Walde. 2017. **Assessing State-of-the-art Sentiment Models on State-of-the-art Sentiment Datasets**. In *Proceedings of the 8th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis*.

## Models
1. Bag-of-Words + L2 regularized Logistic Regression
2. Averaged Embeddings + L2 regularized Logistic Regression
3. Retrofitted Embeddings + L2 regularized Logistic Regression
4. max, min, ave Sentiment Embeddings + L2 regularized Logistic Regression
5. LSTM
6. BiLSTM
7. CNN

## Datasets
1. and 2. [Stanford Sentiment Treebank](http://aclweb.org/anthology/D/D13/D13-1170.pdf) - fine-grained and binary
3. [OpeNER](http://journal.sepln.org/sepln/ojs/ojs/index.php/pln/article/view/4891)
4. [SenTube Auto](https://ikernels-portal.disi.unitn.it/projects/sentube/)
5. [SenTube Tablets](https://ikernels-portal.disi.unitn.it/projects/sentube/)
6. [SemEval 2013 Task 2](https://www.cs.york.ac.uk/semeval-2013/task2.html)

### Requirements

1. Python 3
2. tabulate ```pip install tabulate```
3. sklearn  ```pip install -U scikit-learn```
4. Keras with Theano backend (could work with Tensorflow, but it hasn't been tested)
5. [H5py](http://docs.h5py.org/en/latest/build.html)
6. [Twitter NLP](https://github.com/aritter/twitter_nlp) (included)

### Data you need
1. Word embeddings ([available here](http://www.ims.uni-stuttgart.de/forschung/ressourcen/experiment-daten/sota-sentiment.html))
	- Download and unzip them in directory /embeddings
2. Datasets 	   (provided)


### Running the program

If you want to reproduce the best results for each model reported in the paper, simply clone the repository and run the experiment script:

```
git clone https://github.com/jbarnesspain/sota_sentiment.git
cd sota_sentiment
chmod +x sota_experiment.sh
./sota_experiment.sh
```

### Output

the results will be printed to results/results.txt

the predictions will be kept in /predictions

### Reference

```
@inproceedings{Barnes2017,
  author = {Barnes, Jeremy and Klinger, Roman and Schulte im Walde, Sabine},
  title = {Assessing State-of-the-Art Sentiment Models on State-of-the-Art Sentiment Datasets},
  booktitle = {Proceedings of the 8th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis},
  year = {2017},
  address = {Copenhagen, Denmark}
}
```
