# Assessing State-of-the-art Sentiment Models on State-of-the-art Sentiment Datasets

Jeremy Barnes [jeremy.barnes@upf.edu] / [barnesjy@ims.uni-stuttgart.de]

This experiment runs the best models with the best embeddings
from our work in Barnes et al, 2017. We test 7 different
models on 6 datasets.

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
5. [Twitter NLP](https://github.com/aritter/twitter_nlp) (included)

### Data you need
1. Word embeddings (300-dimensional GoogleNews vectors, 
		    600-dimensional Wikipedia vectors, 
		    600-dimensional retrofitted Wikipedia vectors,
		     50-dimensional Tang twitter sentiment vectors)
2. Datasets 	   (provided here)

Each vector file should have one word vector per line as follows (space delimited):-

```the -1.0 2.4 -0.3 ...```

### Running the program

make sure that sota_experiment is executable (chmod +x sota_experiment).

```./sota_experiment.sh```

### Output

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
