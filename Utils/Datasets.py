import os, re
import numpy as np
from Utils.Representations import *
#from spell_checker import *

class General_Dataset(object):
    """This class takes as input the directory of a corpus annotated for 4 levels
    sentiment. This directory should have 4 .txt files: strneg.txt, neg.txt,
    pos.txt and strpos.txt. It also requires a word embedding model, such as
    those used in word2vec or GloVe.

    binary: instead of 4 classes you have binary (pos/neg). Default is False

    one_hot: the y labels are one hot vectors where the correct class is 1 and
             all others are 0. Default is True.

    dtype: the dtype of the np.array for each vector. Default is np.float32.

    rep: this determines how the word vectors are represented.

         sum_vecs: each sentence is represented by one vector, which is
                    the sum of each of the word vectors in the sentence.

         ave_vecs: each sentence is represented as the average of all of the
                    word vectors in the sentence.

         idx_vecs: each sentence is respresented as a list of word ids given by
                    the word-2-idx dictionary.
    """

    def __init__(self, DIR, model, binary=False, one_hot=True,
                 dtype=np.float32, rep=ave_vecs):

        self.rep = rep
        self.one_hot = one_hot

        Xtrain, Xdev, Xtest, ytrain, ydev, ytest = self.open_data(DIR, model, binary, rep)


        self._Xtrain = Xtrain
        self._ytrain = ytrain
        self._Xdev = Xdev
        self._ydev = ydev
        self._Xtest = Xtest
        self._ytest = ytest
        self._num_examples = len(self._Xtrain)

    def to_array(self, integer, num_labels):
        """quick trick to convert an integer to a one hot vector that
        corresponds to the y labels"""
        integer = integer - 1
        return np.array(np.eye(num_labels)[integer])

    def open_data(self, DIR, model, binary, rep):
        if binary:
            ##################
            # Binary         #
            ##################
            train_neg = getMyData(os.path.join(DIR, 'train/neg.txt'),
                                  0, model, encoding='latin',
                                  representation=rep)
            train_pos = getMyData(os.path.join(DIR, 'train/pos.txt'),
                                  1, model, encoding='latin',
                                  representation=rep)
            dev_neg = getMyData(os.path.join(DIR, 'dev/neg.txt'),
                                0, model, encoding='latin',
                                representation=rep)
            dev_pos = getMyData(os.path.join(DIR, 'dev/pos.txt'),
                                1, model, encoding='latin',
                                representation=rep)
            test_neg = getMyData(os.path.join(DIR, 'test/neg.txt'),
                                 0, model, encoding='latin',
                                 representation=rep)
            test_pos = getMyData(os.path.join(DIR, 'test/pos.txt'),
                                 1, model, encoding='latin',
                                 representation=rep)

            traindata = train_pos + train_neg
            devdata = dev_pos + dev_neg
            testdata = test_pos + test_neg
            # Training data
            Xtrain = [data for data, y in traindata]
            if self.one_hot is True:
                ytrain = [self.to_array(y, 2) for data, y in traindata]
            else:
                ytrain = [y for data, y in traindata]

            # Dev data
            Xdev = [data for data, y in devdata]
            if self.one_hot is True:
                ydev = [self.to_array(y, 2) for data, y in devdata]
            else:
                ydev = [y for data, y in devdata]

            # Test data
            Xtest = [data for data, y in testdata]
            if self.one_hot is True:
                ytest = [self.to_array(y, 2) for data, y in testdata]
            else:
                ytest = [y for data, y in testdata]
        
        else:
            ##################
            # 4 CLASS        #
            ##################
            train_strneg = getMyData(os.path.join(DIR, 'train/strneg.txt'),
                                  0, model, encoding='latin',
                                  representation=rep)
            train_strpos = getMyData(os.path.join(DIR, 'train/strpos.txt'),
                                  3, model, encoding='latin',
                                  representation=rep)
            train_neg = getMyData(os.path.join(DIR, 'train/neg.txt'),
                                  1, model, encoding='latin',
                                  representation=rep)
            train_pos = getMyData(os.path.join(DIR, 'train/pos.txt'),
                                  2, model, encoding='latin',
                                  representation=rep)
            dev_strneg = getMyData(os.path.join(DIR, 'dev/strneg.txt'),
                                0, model, encoding='latin',
                                representation=rep)
            dev_strpos = getMyData(os.path.join(DIR, 'dev/strpos.txt'),
                                3, model, encoding='latin',
                                representation=rep)
            dev_neg = getMyData(os.path.join(DIR, 'dev/neg.txt'),
                                1, model, encoding='latin',
                                representation=rep)
            dev_pos = getMyData(os.path.join(DIR, 'dev/pos.txt'),
                                2, model, encoding='latin',
                                representation=rep)
            test_strneg = getMyData(os.path.join(DIR, 'test/strneg.txt'),
                                 0, model, encoding='latin',
                                 representation=rep)
            test_strpos = getMyData(os.path.join(DIR, 'test/strpos.txt'),
                                 3, model, encoding='latin',
                                 representation=rep)
            test_neg = getMyData(os.path.join(DIR, 'test/neg.txt'),
                                 1, model, encoding='latin',
                                 representation=rep)
            test_pos = getMyData(os.path.join(DIR, 'test/pos.txt'),
                                 2, model, encoding='latin',
                                 representation=rep)

            traindata = train_pos + train_neg + train_strneg + train_strpos
            devdata = dev_pos + dev_neg + dev_strneg + dev_strpos
            testdata = test_pos + test_neg + test_strneg + test_strpos


            # Training data
            Xtrain = [data for data, y in traindata]
            if self.one_hot is True:
                ytrain = [self.to_array(y, 4) for data, y in traindata]
            else:
                ytrain = [y for data, y in traindata]

            # Dev data
            Xdev = [data for data, y in devdata]
            if self.one_hot is True:
                ydev = [self.to_array(y, 4) for data, y in devdata]
            else:
                ydev = [y for data, y in devdata]

            # Test data
            Xtest = [data for data, y in testdata]
            if self.one_hot is True:
                ytest = [self.to_array(y, 4) for data, y in testdata]
            else:
                ytest = [y for data, y in testdata]

        if self.rep is not words:
            Xtrain = np.array(Xtrain)
            Xdev = np.array(Xdev)
            Xtest = np.array(Xtest)
        ytrain = np.array(ytrain)
        ydev = np.array(ydev)
        ytest = np.array(ytest)

        return Xtrain, Xdev, Xtest, ytrain, ydev, ytest

#########################################################################
# Other Variations that make it easier                                  #
#########################################################################

class Catalan_Dataset(General_Dataset):
    def __init__(self, model, binary=False, one_hot=True,
             dtype=np.float32, rep=ave_vecs):
        DIR = '/home/jeremy/NS/Keep/Permanent/Corpora/OpeNER_Data/catalan_split/'
        super().__init__(DIR, model, binary, one_hot, dtype, rep)

class English_Dataset(General_Dataset):
    def __init__(self, model, binary=False, one_hot=True,
             dtype=np.float32, rep=ave_vecs):
        DIR = '../datasets/opener'
        super().__init__(DIR, model, binary, one_hot, dtype, rep)

class Spanish_Dataset(General_Dataset):
    def __init__(self, model, binary=False, one_hot=True,
             dtype=np.float32, rep=ave_vecs):
        DIR = '/home/jeremy/NS/Keep/Permanent/Corpora/OpeNER_Data/spanish_split/'
        super().__init__(DIR, model, binary, one_hot, dtype, rep)


#########################################################################
# Rotten Tomatoes Dataset: Pang and Lee, (2005).                        #
#########################################################################


class Sentence_Polarity_Dataset(object):
    """This class takes as input the directory of the rotten tomatoes
    dataset, divided into pos.txt and neg.txt. It also requires a word 
    embedding model, such as those used in word2vec or GloVe or a 
    word-to-index dictionary.

    one_hot: the y labels are one hot vectors where the correct class is 1 and
             all others are 0. Default is True.

    dtype: the dtype of the np.array for each vector. Default is np.float32.

    rep: this determines how the word vectors are represented.

         sum_vecs: each sentence is represented by one vector, which is
                    the sum of each of the word vectors in the sentence.

         ave_vecs: each sentence is represented as the average of all of the
                    word vectors in the sentence.

         idx_vecs: each sentence is respresented as a list of word ids given by
                    the word-2-idx dictionary.
    """

    def __init__(self, DIR, model, one_hot=True,
                 dtype=np.float32, rep=ave_vecs):

        self.rep = rep
        self.one_hot = one_hot
        Xtrain, Xdev, Xtest, ytrain, ydev, ytest = self.open_data(DIR, model, rep)


        self._Xtrain = Xtrain
        self._ytrain = ytrain
        self._Xdev = Xdev
        self._ydev = ydev
        self._Xtest = Xtest
        self._ytest = ytest
        self._num_examples = len(self._Xtrain)

    def open_data(self, DIR, model, rep):
        train_neg = getMyData(os.path.join(DIR, 'train/neg.txt'),
                              0, model, encoding='latin',
                              representation=rep)
        train_pos = getMyData(os.path.join(DIR, 'train/pos.txt'),
                              1, model, encoding='latin',
                              representation=rep)
        dev_neg = getMyData(os.path.join(DIR, 'dev/neg.txt'),
                            0, model, encoding='latin',
                            representation=rep)
        dev_pos = getMyData(os.path.join(DIR, 'dev/pos.txt'),
                            1, model, encoding='latin',
                            representation=rep)
        test_neg = getMyData(os.path.join(DIR, 'test/neg.txt'),
                             0, model, encoding='latin',
                             representation=rep)
        test_pos = getMyData(os.path.join(DIR, 'test/pos.txt'),
                             1, model, encoding='latin',
                             representation=rep)

        traindata = train_pos + train_neg
        devdata = dev_pos + dev_neg
        testdata = test_pos + test_neg

        Xtrain = [data for data, y in traindata]
        if self.one_hot is True:
            ytrain = [self.to_array(y, 2) for data, y in traindata]
        else:
            ytrain = [y for data, y in traindata]
        Xtrain = np.array(Xtrain)
        ytrain = np.array(ytrain)

        Xdev = [data for data, y in devdata]
        if self.one_hot is True:
            ydev = [self.to_array(y, 2) for data, y in devdata]
        else:
            ydev = [y for data, y in devdata]

        Xtest = [data for data, y in testdata]
        if self.one_hot is True:
            ytest = [self.to_array(y, 2) for data, y in testdata]
        else:
            ytest = [y for data, y in testdata]

        if self.rep is not words:
            Xtrain = np.array(Xtrain)
            Xdev = np.array(Xdev)
            Xtest = np.array(Xtest)

        ytrain = np.array(ytrain)
        ydev = np.array(ydev)
        ytest = np.array(ytest)

        return Xtrain, Xdev, Xtest, ytrain, ydev, ytest

    def to_array(self, integer, num_labels):
        """quick trick to convert an integer to a one hot vector that
        corresponds to the y labels"""
        integer = integer - 1
        return np.array(np.eye(num_labels)[integer])

    def finish_tokenizing(self, tokens):
        semi_tokens = ' '.join(tokens)
        re_tokenized = re.sub("'", " ' ", semi_tokens)
        re_tokenized = re.sub('-', ' ', re_tokenized)
        re_tokenized = re.sub('/', ' ', re_tokenized)
        return re_tokenized.split()


class Concat_Sentence_Polarity_Dataset(Sentence_Polarity_Dataset):

    def __init__(self, DIR, model, model2, one_hot=True,
                 dtype=np.float32, rep=ave_vecs):

        self.rep = rep
        self.one_hot = one_hot

        Xtrain, Xdev, Xtest, ytrain, ydev, ytest = self.open_data(DIR, model, rep)
        Xtrain2, Xdev2, Xtest2, ytrain2, ydev2, ytest2 = self.open_data(DIR, model2, rep)

        self.CONCX_train = self.concatenate(Xtrain, Xtrain2)
        self.CONCX_dev = self.concatenate(Xdev, Xdev2)
        self.CONCX_test = self.concatenate(Xtest, Xtest2)

        self._Xtrain = Xtrain
        self._ytrain = ytrain
        self._Xdev = Xdev
        self._ydev = ydev
        self._Xtest = Xtest
        self._ytest = ytest
        self._Xtrain2 = Xtrain2
        self._ytrain2 = ytrain2
        self._Xdev2 = Xdev2
        self._ydev2 = ydev2
        self._Xtest2 = Xtest2
        self._ytest2 = ytest2
        self._num_examples = len(self._Xtrain)

    def concatenate(self, x1, x2):
        return np.concatenate((x1, x2), axis=1)

class Concat_Opener_Dataset(Concat_Sentence_Polarity_Dataset):
    """Concatenated Opener Dataset:
    See Concat_Sentence_Polarity_Dataset for details."""

    def open_data(self, DIR, model, rep):
        datastrneg = getMyData(os.path.join(DIR, 'strneg.txt'), 1, model,
                               representation=rep)
        dataneg = getMyData(os.path.join(DIR, 'neg.txt'), 2, model,
                            representation=rep)
        datapos = getMyData(os.path.join(DIR, 'pos.txt'), 3, model,
                            representation=rep)
        datastrpos = getMyData(os.path.join(DIR, 'strpos.txt'), 4, model,
                               representation=rep)
        traindata = datastrneg[:int((len(datastrneg) * .75))]+ dataneg[:int((len(dataneg) * .75))] + datapos[:int((len(datapos) * .75))] + datastrpos[:int((len(datastrpos) * .75))]
        devdata = datastrneg[int((len(datastrneg) * .75)):int((len(dataneg) * .85))]+ dataneg[int((len(dataneg) * .75)):int((len(dataneg) * .85))]+ datapos[int((len(datapos) * .75)):int((len(dataneg) * .85))]+ datastrpos[int((len(datastrpos) * .75)):int((len(dataneg) * .85))]
        testdata = datastrneg[int((len(datastrneg) * .85)):]+ dataneg[int((len(dataneg) * .85)):]+ datapos[int((len(datapos) * .85)):]+ datastrpos[int((len(datastrpos) * .85)):]

        Xtrain = [data for data, y in traindata]
        if self.one_hot is True:
            ytrain = [self.to_array(y, 4) for data, y in traindata]
        else:
            ytrain = [y for data, y in traindata]

        Xdev = [data for data, y in devdata]
        if self.one_hot is True:
            ydev = [self.to_array(y, 4) for data, y in devdata]
        else:
            ydev = [y for data, y in devdata]

        Xtest = [data for data, y in testdata]
        if self.one_hot is True:
            ytest = [self.to_array(y, 4) for data, y in testdata]
        else:
            ytest = [y for data, y in testdata]

        if self.rep is not words:
            Xtrain = np.array(Xtrain)
            Xdev = np.array(Xdev)
            Xtest = np.array(Xtest)

        ytrain = np.array(ytrain)
        ydev = np.array(ydev)
        ytest = np.array(ytest)


        return Xtrain, Xdev, Xtest, ytrain, ydev, ytest

class Stanford_Sentiment_Dataset(object):
    """Stanford Sentiment Treebank
    """

    def __init__(self, DIR, model, one_hot=True,
                 dtype=np.float32, binary=False, rep=ave_vecs):

        self.rep = rep
        self.one_hot = one_hot
        self.binary = binary

        Xtrain, Xdev, Xtest, ytrain, ydev, ytest = self.open_data(DIR, model, rep)

        self._Xtrain = Xtrain
        self._ytrain = ytrain
        self._Xdev = Xdev
        self._ydev = ydev
        self._Xtest = Xtest
        self._ytest = ytest
        self._num_examples = len(self._Xtrain)

    def flatten(self, tree_sent):
        label = int(tree_sent[1])
        text = re.sub('\([0-9]', ' ', tree_sent).replace(')','').split()
        return label, ' '.join(text)

    def to_array(self, integer, num_labels):
        """quick trick to convert an integer to a one hot vector that
        corresponds to the y labels"""
        integer = integer - 1
        return np.array(np.eye(num_labels)[integer])

    def remove_neutral(self, data):
        final = []
        for y, x in data:
            if y in [0, 1]:
                final.append((0, x))
            elif y in [3, 4]:
                final.append((1, x))
        return final

    def open_data(self, DIR, model, rep):
        train = open(os.path.join(DIR, 'train.txt'))
        dev = open(os.path.join(DIR, 'dev.txt'))
        test = open(os.path.join(DIR, 'test.txt'))

        train_data = [self.flatten(x) for x in train]
        if self.binary:
            train_data = self.remove_neutral(train_data)
        ytrain, Xtrain = zip(*train_data)
        Xtrain = [rep(sent, model) for sent in Xtrain]

        dev_data = [self.flatten(x) for x in dev]
        if self.binary:
            dev_data = self.remove_neutral(dev_data)
        ydev, Xdev = zip(*dev_data)
        Xdev = [rep(sent, model) for sent in Xdev]

        test_data = [self.flatten(x) for x in test]
        if self.binary:
            test_data = self.remove_neutral(test_data)
        ytest, Xtest = zip(*test_data)
        Xtest = [rep(sent, model) for sent in Xtest]

        if self.one_hot is True:
            if self.binary:
                ytrain = [self.to_array(y, 2) for y in ytrain]
                ydev = [self.to_array(y, 2) for y in ydev]
                ytest = [self.to_array(y, 2) for y in ytest]
            else:
                ytrain = [self.to_array(y, 5) for y in ytrain]
                ydev = [self.to_array(y, 5) for y in ydev]
                ytest = [self.to_array(y, 5) for y in ytest]


        if self.rep is not words:
            Xtrain = np.array(Xtrain)
            Xdev = np.array(Xdev)
            Xtest = np.array(Xtest)

        ytrain = np.array(ytrain)
        ydev = np.array(ydev)
        ytest = np.array(ytest)

        return Xtrain, Xdev, Xtest, ytrain, ydev, ytest

class Twitter_Sentiment_Dataset(object):
    """
    """

    def __init__(self, file, model, one_hot=True,
                 dtype=np.float32, rep=ave_vecs):

        self.rep = rep
        self.one_hot = one_hot
        Xtrain, Xdev, Xtest, ytrain, ydev, ytest = self.open_data(file, model, rep)

        self._Xtrain = Xtrain
        self._ytrain = ytrain
        self._Xdev = Xdev
        self._ydev = ydev
        self._Xtest = Xtest
        self._ytest = ytest
        self._num_examples = len(self._Xtrain)

    def to_array(self, integer, num_labels):
        """quick trick to convert an integer to a one hot vector that
        corresponds to the y labels"""
        integer = integer - 1
        return np.array(np.eye(num_labels)[integer])

    def process_line(self, line):
        split = line.split(',')
        label = int(split[1])
        text = split[-1].strip()
        return label, text

    def open_data(self, file, model, rep):
        
        file = open(file)
        file.readline() # first line is just headers
        data = [self.process_line(line) for line in file]

        num_examples = len(data)
        train_idx = 50000 #int(num_examples * .75)
        dev_idx = 60000 #int(num_examples * .8)
        test_idx  = 80000
        
        train_data = data[:train_idx]
        dev_data = data[train_idx: dev_idx]
        test_data = data[dev_idx:test_idx]
        
        ytrain, Xtrain = zip(*train_data)
        Xtrain = [rep(sent, model) for sent in Xtrain]
        
        ydev, Xdev = zip(*dev_data)
        Xdev = [rep(sent, model) for sent in Xdev]

        ytest, Xtest = zip(*test_data)
        Xtest = [rep(sent, model) for sent in Xtest]

        if self.one_hot is True:
            ytrain = [self.to_array(y, 2) for y in ytrain]
            ydev = [self.to_array(y, 2) for y in ydev]
            ytest = [self.to_array(y, 2) for y in ytest]


        if self.rep is not words:
            Xtrain = np.array(Xtrain)
            Xdev = np.array(Xdev)
            Xtest = np.array(Xtest)

        ytrain = np.array(ytrain)
        ydev = np.array(ydev)
        ytest = np.array(ytest)

        return Xtrain, Xdev, Xtest, ytrain, ydev, ytest


##########################################################################

class Amazon_Dataset(General_Dataset):
    
    def open_data(self, DIR, model, binary, rep):
        
        neg = open(os.path.join(DIR,'negative.review')).read()
        pos = open(os.path.join(DIR,'positive.review')).read()

        pos = pos.split('<review>')[1:]
        neg = neg.split('<review>')[1:]

        posX = [self.get_between(l, '<review_text>\n', '\n</review_text>') for l in pos]
        negX = [self.get_between(l, '<review_text>\n', '\n</review_text>') for l in neg]

        if binary:
            posy = [1] * len(posX)
            negy = [0] * len(negX)
            if self.one_hot is True:
            	posy = [self.to_array(y, 2) for y in posy]
            	negy = [self.to_array(y, 2) for y in negy]
        else:
            posy = [float(self.get_between(l, '<rating>\n', '\n</rating>')) for l in pos]
            negy = [float(self.get_between(l, '<rating>\n', '\n</rating>')) for l in neg]
            posy = [self.change_y(y) for y in posy]
            negy = [self.change_y(y) for y in negy]
            if self.one_hot is True:
            	posy = [self.to_array(y, 4) for y in posy]
            	negy = [self.to_array(y, 4) for y in negy]

        pos = list(zip(posy, posX))
        neg = list(zip(negy, negX))
        
        train_idx = int(len(pos) * .75)
        dev_idx = int(len(pos) * .8)

        train_neg = neg[:train_idx]
        dev_neg = neg[train_idx:dev_idx]
        test_neg = neg[dev_idx:]

        train_pos = pos[:train_idx]
        dev_pos = pos[train_idx:dev_idx]
        test_pos = pos[dev_idx:]

        train_data = train_pos + train_neg
        dev_data = dev_pos + dev_neg
        test_data = test_pos + test_neg

        ytrain, Xtrain = zip(*train_data)
        Xtrain = [rep(sent, model) for sent in Xtrain]
        
        ydev, Xdev = zip(*dev_data)
        Xdev = [rep(sent, model) for sent in Xdev]

        ytest, Xtest = zip(*test_data)
        Xtest = [rep(sent, model) for sent in Xtest]
        

        if self.rep is not words:
            Xtrain = np.array(Xtrain)
            Xdev = np.array(Xdev)
            Xtest = np.array(Xtest)
            
        ytrain = np.array(ytrain)
        ydev = np.array(ydev)
        ytest = np.array(ytest)

        return Xtrain, Xdev, Xtest, ytrain, ydev, ytest

    def get_between(self, x, l, r):
        mid = x.split(l)[1]
        return mid.split(r)[0]

    def change_y(self, y):
    	if y == 1.0:
    		return 0
    	elif y == 2.0:
    		return 1
    	elif y == 4.0:
    		return 2
    	elif y == 5.0:
    		return 3


###################################################

class Book_Dataset(Amazon_Dataset):
    def __init__(self, model, binary=False, one_hot=True,
             dtype=np.float32, rep=ave_vecs):
        DIR = '../datasets/amazon-multi-domain/books'
        super(Amazon_Dataset, self).__init__(DIR, model, binary, one_hot, dtype, rep)

class DVD_Dataset(Amazon_Dataset):
    def __init__(self, model, binary=False, one_hot=True,
             dtype=np.float32, rep=ave_vecs):
        DIR = '../datasets/amazon-multi-domain/dvd'
        super(Amazon_Dataset, self).__init__(DIR, model, binary, one_hot, dtype, rep)

class Electronics_Dataset(Amazon_Dataset):
    def __init__(self, model, binary=False, one_hot=True,
             dtype=np.float32, rep=ave_vecs):
        DIR = '../datasets/amazon-multi-domain/electronics'
        super(Amazon_Dataset, self).__init__(DIR, model, binary, one_hot, dtype, rep)

class Kitchen_Dataset(Amazon_Dataset):
    def __init__(self, model, binary=False, one_hot=True,
             dtype=np.float32, rep=ave_vecs):
        DIR = '../datasets/amazon-multi-domain/kitchen_&_housewares'
        super(Amazon_Dataset, self).__init__(DIR, model, binary, one_hot, dtype, rep)