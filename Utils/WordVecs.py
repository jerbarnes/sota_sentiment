import numpy as np
import pickle
from scipy.spatial.distance import cosine

class WordVecs(object):
    """Import word2vec files saved in txt format.
    Creates an embedding matrix and two dictionaries
    (1) a word to index dictionary which returns the index
    in the embedding matrix
    (2) a index to word dictionary which returns the word
    given an index.
    """


    def __init__(self, file, file_type='word2vec', vocab=None):
        self.file_type = file_type
        self.vocab = vocab
        (self.vocab_length, self.vector_size, self._matrix,
         self._w2idx, self._idx2w) = self._read_vecs(file)

    def __getitem__(self, y):
        try:
            return self._matrix[self._w2idx[y]]
        except KeyError:
            raise KeyError
        except IndexError:
            raise IndexError
        

    def _read_vecs(self, file):
        """Assumes that the first line of the file is
        the vocabulary length and vector dimension."""

        if self.file_type == 'word2vec':
            txt = open(file).readlines()
            vocab_length, vec_dim = [int(i) for i in txt[0].split()]
            txt = txt[1:]
        elif self.file_type == 'bin':
            txt = open(file, 'rb')
            header = txt.readline()
            vocab_length, vec_dim = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * vec_dim

        else:
            txt = open(file).readlines()
            vocab_length = len(txt)
            vec_dim = len(txt[0].split()[1:])


        if self.vocab:
            emb_matrix = np.zeros((len(self.vocab), vec_dim))
            vocab_length = len(self.vocab)
        else:
            emb_matrix = np.zeros((vocab_length, vec_dim))
        w2idx = {}

        # Read a binary file
        if self.file_type == 'bin':
            for line in range(vocab_length):
                word = []
                while True:
                    ch = txt.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                # if you have vocabulary, you can only load these words
                if self.vocab:
                    if word in self.vocab:
                        w2idx[word] = len(w2idx)
                        emb_matrix[w2idx[word]] = np.fromstring(txt.read(binary_len), dtype='float32')  
                    else:
                        txt.read(binary_len)
                else:
                    w2idx[word] = len(w2idx)
                    emb_matrix[w2idx[word]] = np.fromstring(txt.read(binary_len), dtype='float32')  

        # Read a txt file
        else:    
            for item in txt:
                if self.file_type == 'tang':            # tang separates with tabs
                    split = item.strip().replace(',','.').split()
                else:
                    split = item.strip().split(' ')
                try:
                    word, vec = split[0], np.array(split[1:], dtype=float)

                    # if you have vocabulary, only load these words
                    if self.vocab:
                        if word in self.vocab:
                            w2idx[word] = len(w2idx)
                            emb_matrix[w2idx[word]] = vec
                        else:
                            pass
                    else:
                        if len(vec) == vec_dim:
                            w2idx[word] = len(w2idx)
                            emb_matrix[w2idx[word]] = vec
                        else:
                            pass
                except ValueError:
                    pass

            
        idx2w = dict([(i, w) for w, i in w2idx.items()])

        return vocab_length, vec_dim, emb_matrix, w2idx, idx2w

    def most_similar(self, word, num_similar=5):
        idx = self._w2idx[word]
        y = list(range(self._matrix.shape[0]))
        y.pop(idx)
        most_similar = [(1,0)] * num_similar
        for i in y:
            dist = 0
            dist = cosine(self._matrix[idx], self._matrix[i])
            if dist < most_similar[-1][0]:
                most_similar.pop()
                most_similar.append((dist,i))
                most_similar = sorted(most_similar)
        most_similar = [(distance, self._idx2w[i]) for (distance, i) in most_similar]
        return most_similar

    def normalize(self):
        row_sums = self._matrix.sum(axis=1, keepdims=True)
        self._matrix = self._matrix / row_sums



class GloveVecs(object):

    def __init__(self, file, vector_size):
        self.word_to_vec = self.read_vecs(file)
        self.vector_size = vector_size
        self.vocab_length = len(self.word_to_vec)

    def __getitem__(self,y):
        return self.word_to_vec.get(y)
    
    def read_vecs(self, file):
        txt = open(file).readlines()
        word_to_vec = {}
        lines_read = 0
        for item in txt:
            try:
                split = item.split()
                word, vec = split[0], np.array(split[1:], dtype='float32')
                word_to_vec[word] = vec
                lines_read += 1
                if lines_read % 100 == 0:
                    self.drawProgressBar((lines_read/len(txt)))
            except NameError:
                continue
            except SyntaxError:
                continue
            except ValueError:
                continue

        return word_to_vec

    def most_similar(self, word, num_similar=5):
        idx = self._w2idx[word]
        y = list(range(self._matrix.shape[0]))
        y.pop(idx)
        most_similar = [(1,0)] * num_similar
        for i in y:
            dist = 0
            dist = cosine(self._matrix[idx], self._matrix[i])
            if dist < most_similar[-1][0]:
                most_similar.pop()
                most_similar.append((dist,i))
                most_similar = sorted(most_similar)
        most_similar = [(distance, self._idx2w[i]) for (distance, i) in most_similar]
        return most_similar

    def normalize(self):
        row_sums = self._matrix.sum(axis=1, keepdims=True)
        self._matrix = self._matrix / row_sums


    
    def drawProgressBar(self, percent, barLen = 20):
        sys.stdout.write("\r")
        progress = ""
        for i in range(barLen):
            if i < int(barLen * percent):
                progress += "="
            else:
                progress += " "
        sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
        sys.stdout.flush()


class ConcatVecs(WordVecs):
    
    def __init__(self, file, file2, vocab_file, concat_vec_dim):
        self.file = file
        self.file2 = file2
        (self.vocab_length, self.vector_size, self._matrix,
         self._w2idx, self._idx2w) = self._read_dual_vecs(file,
                                                          file2,
                                                          vocab_file,
                                                          concat_vec_dim)

    def _read_dual_vecs(self, file1, file2, vocab_file, concat_vec_dim):
        
        with open(vocab_file, 'rb') as vfile:
            w2idx = pickle.load(vfile)

        emb_matrix = np.zeros((len(w2idx), concat_vec_dim))

        idx2w = dict([(i, w) for w, i in w2idx.items()])

        vecs1 = WordVecs(file1)
        vecs1.normalize()
        vecs1_vocab = set(vecs1._w2idx.keys())
        vecs2 = WordVecs(file2)
        vecs2.normalize()
        vecs2_vocab = set(vecs2._w2idx.keys())

        for word, i in w2idx.items():
            if word in vecs1_vocab:
                vec1 = vecs1[word]
            else:
                vec1 = np.zeros((int(concat_vec_dim / 2)))
            if word in vecs2_vocab:
                vec2 = vecs2[word]
            else:
                vec2 = np.zeros((int(concat_vec_dim / 2)))
            concat = np.concatenate((vec1, vec2))
            emb_matrix[i] = concat

        return len(idx2w), concat_vec_dim, emb_matrix, w2idx, idx2w

