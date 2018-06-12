import numpy as np
import pandas as pd
import sys

from gensim.models import Word2Vec


class Tool():
    
    def __init__(self):
        self._initialize()
    
    def _initialize(self):

        np.random.seed(0)
        df = pd.read_pickle('./data/df')
        idx = np.array(df.index)
        np.random.shuffle(idx)
        df = df.loc[idx]
        self.df = df[df['label'] != 'neu']
        
        model = Word2Vec.load('./data/w2v_model')
        self.i2w = model.wv.index2word
        
        self.table = pd.read_pickle('./data/lookup_table')

    def set_hyper_parameters(self, batch_size):

        df = self.df
        
        self.batch_size = batch_size
        self.n_batch = len(df) // batch_size
        self.n_epoch = 0
    
    def get_batch(self, i):
    
        n_batch = self.n_batch
        batch_size = self.batch_size
        df = self.df
    
        def get_X(sen):
            X = np.zeros(64, dtype=np.int32)-1
            X[-len(sen):] = sen
            return X

        def get_y(label):
            idx = ['neg', 'pos'].index(label)
            y = np.array([0, 0])
            y[idx] = 1
            return y
        
        def get_y_isreal(label):
            idx = ['fake', 'real'].index(label)
            y = np.array([0, 0])
            y[idx] = 1
            return y
               
        start, end = i*batch_size, (i+1)*batch_size
        df = df[start:end]
        
        Xs = np.array([get_X(sen) for sen in df['pos']])
        ys = np.array([get_y(label) for label in df['label']])
        y_valids = np.array([get_y_isreal(label) for label in df['valid']])
        
        if n_batch == i+1:
            self.n_epoch += 1

        return Xs, ys, y_valids
    
    def convet_a2s(self, array):

        i2w = self.i2w
        table = self.table
        
        def get_word(i, v):
            idx = np.linalg.norm(table-v, axis=1).argmin()
            if idx == len(i2w):
                word = ''
            else:
                word = i2w[idx]
                word = word.split("-")[0]
                word = word+" "
            sys.stdout.write("\r%5.2f%%" % ((i+1)/len(array)*100))
            return word

        sen = ''.join([get_word(i, v) for i, v in enumerate(array)])
        print()
        sen = sen[:-1]

        return sen