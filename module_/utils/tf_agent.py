import numpy as np
import pandas as pd
import sys

from gensim.models import Word2Vec
import tensorflow as tf

class TF_Agent():
    
    def __init__(self):
        self._initialize()
    
    def _initialize(self):

        tf.set_random_seed(0)
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

    def get_graph(self):
        
        table = self.table

        tf.reset_default_graph()
        graph = tf.get_default_graph()

        X = tf.placeholder(dtype=tf.int32, shape=[None, 64])
        y = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        y_valid = tf.placeholder(dtype=tf.float32, shape=[None, 2])

        lookup_table = tf.Variable(table, dtype=tf.float32)
        X_ebd = tf.nn.embedding_lookup(lookup_table, X)

        lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(128)
        lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(128)
        output, _ = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell_fw, lstm_cell_bw, X_ebd, dtype=tf.float32
        )
        logit = tf.concat(output, axis=1)

        ############ for lebal ##########
        conv = tf.reshape(tf.layers.conv1d(logit, 128, 128), [-1, 128])
        conv_bn = tf.layers.batch_normalization(conv)
        conv_sg = tf.nn.sigmoid(conv_bn)
        conv_fin = tf.layers.dropout(conv_sg, 0.2)

        w = tf.Variable(tf.truncated_normal([128, 2], dtype=tf.float32))
        b = tf.Variable(tf.truncated_normal([2], dtype=tf.float32))
        fin = tf.nn.softmax(tf.matmul(conv_fin, w)+b)

        acc_label = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(fin, axis=1), tf.argmax(y, axis=1)),
            dtype=tf.float32
        ))
        loss_label = tf.losses.softmax_cross_entropy(y, fin)
        ############ for lebal ##########

        ########### for valid ###########
        conv_valid = tf.reshape(tf.layers.conv1d(logit, 128, 128), [-1, 128])
        conv_bn_valid = tf.layers.batch_normalization(conv_valid)
        conv_sg_valid = tf.nn.sigmoid(conv_bn_valid)
        conv_fin_valid = tf.layers.dropout(conv_sg_valid, 0.2)

        w_valid = tf.Variable(tf.truncated_normal([128, 2], dtype=tf.float32))
        b_valid = tf.Variable(tf.truncated_normal([2], dtype=tf.float32))
        fin_valid = tf.nn.softmax(tf.matmul(conv_fin_valid, w_valid)+b_valid)

        acc_valid = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(fin_valid, axis=1), tf.argmax(y_valid, axis=1)),
            dtype=tf.float32
        ))
        loss_valid = tf.losses.softmax_cross_entropy(y_valid, fin_valid)
        ########### for valid ###########

        ############## for gen ##########
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=512)
        output_gen, _ = tf.nn.dynamic_rnn(
            lstm_cell, logit, dtype=tf.float32
        )

        w_gen = tf.Variable(tf.truncated_normal([64, 128], dtype=tf.float32))
        b_gen = tf.Variable(tf.truncated_normal([64, 512], dtype=tf.float32))
        fin_gen = tf.map_fn(lambda x: tf.matmul(w_gen, x)+b_gen, output_gen)

        loss_gen = tf.losses.mean_squared_error(X_ebd, fin_gen)
        ############## for gen ##########

        loss = loss_label+loss_valid+loss_gen
        learn = tf.train.AdamOptimizer().minimize(loss)
        loss_total = loss

        initilizer = tf.global_variables_initializer()

        return graph