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
            X[:len(sen)] = sen
            return X

        def get_y_label(label):
            idx = ['neg', 'pos'].index(label)
            y = np.array([0, 0])
            y[idx] = 1
            return y
        
        def get_y_valid(label):
            idx = ['fake', 'real'].index(label)
            y = np.array([0, 0])
            y[idx] = 1
            return y
               
        start, end = i*batch_size, (i+1)*batch_size
        df = df[start:end]
        
        Xs = np.array([get_X(sen) for sen in df['pos']])
        y_labels = np.array([get_y_label(label) for label in df['label']])
        y_valids = np.array([get_y_valid(label) for label in df['valid']])
        
        if n_batch == i+1:
            self.n_epoch += 1

        return Xs, y_labels, y_valids
    
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
            sys.stdout.write("\r%5.2f%%" % ((i+1)/len(array)*100))
            return word

        sen = ' '.join([get_word(i, v) for i, v in enumerate(array)])
        print()

        return sen

    def convet_l2s(self, l):
        i2w = self.i2w    
        sen = ' '.join([i2w[i].split("-")[0] for i in l])
        return sen

    def init_graph(self):
        
        table = self.table

        tf.reset_default_graph()
        self.graph = tf.get_default_graph()

        X = tf.placeholder(dtype=tf.int32, shape=[None, 64])
        y_label = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        y_valid = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        style = tf.placeholder(dtype=tf.float32, shape=[None, 2])

        self.X = X
        self.y_label = y_label
        self.y_valid = y_valid
        self.style = style

        lookup_table = tf.Variable(table, dtype=tf.float32)
        X_ebd = tf.nn.embedding_lookup(lookup_table, X)

        lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(256)
        lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(256)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell_fw, lstm_cell_bw, X_ebd, dtype=tf.float32
        )
        output = tf.concat(outputs, axis=1)

        conv = tf.reshape(tf.layers.conv1d(output, 256, 128), [-1, 256])
        conv_bn = tf.layers.batch_normalization(conv)
        conv_sg = tf.nn.sigmoid(conv_bn)
        latent = tf.layers.dropout(conv_sg, 0.2)

        ############ for wake ##########
        w0_label = tf.Variable(tf.truncated_normal([256, 64],
                                                dtype=tf.float32))
        b0_label = tf.Variable(tf.truncated_normal([64],
                                                dtype=tf.float32))
        a0_label = tf.nn.softmax(tf.matmul(latent, w0_label)+b0_label)
        d0_label = tf.layers.dropout(a0_label, 0.2)
        w1_label = tf.Variable(tf.truncated_normal([64, 2],
                                                dtype=tf.float32))
        b1_label = tf.Variable(tf.truncated_normal([2],
                                                dtype=tf.float32))
        a1_label = tf.nn.softmax(tf.matmul(d0_label, w1_label)+b1_label)

        w0_valid = tf.Variable(tf.truncated_normal([256, 64],
                                                dtype=tf.float32))
        b0_valid = tf.Variable(tf.truncated_normal([64],
                                                dtype=tf.float32))
        a0_valid = tf.nn.softmax(tf.matmul(latent, w0_valid)+b0_valid)
        d0_valid = tf.layers.dropout(a0_valid, 0.2)
        w1_valid = tf.Variable(tf.truncated_normal([64, 2],
                                                dtype=tf.float32))
        b1_valid = tf.Variable(tf.truncated_normal([2],
                                                dtype=tf.float32))
        a1_valid = tf.nn.softmax(tf.matmul(d0_valid, w1_valid)+b1_valid)

        loss_label = tf.losses.softmax_cross_entropy(y_label, a1_label)        
        loss_valid = tf.losses.softmax_cross_entropy(y_valid, a1_valid)

        self.acc_label = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(a1_label, axis=1), tf.argmax(y_label, axis=1)),
            dtype=tf.float32
        ))
        self.acc_valid = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(a1_valid, axis=1), tf.argmax(y_valid, axis=1)),
            dtype=tf.float32
        ))
        ############ for wake ##########

        ############ for sleep #########
        output_l = tf.concat([
            tf.reshape(latent, shape=[-1, 1, 256]),
            output
        ], axis=1)

        output_s = tf.concat([
            output_l,
            tf.transpose(tf.zeros(shape=[129, 128, 2])+style, [1, 0, 2])
        ], axis=2)

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=512)
        output_gen, _ = tf.nn.dynamic_rnn(
            lstm_cell, output_s, dtype=tf.float32
        )

        w_gen = tf.Variable(tf.truncated_normal([64, 128+1], dtype=tf.float32))
        b_gen = tf.Variable(tf.truncated_normal([64, 512], dtype=tf.float32))
        f_gen = tf.map_fn(lambda x: tf.matmul(w_gen, x)+b_gen, output_gen)
        self.sentence = f_gen

        loss_gen = tf.losses.mean_squared_error(X_ebd, f_gen)
        self.acc_gen = loss_gen
        ############ for sleep #########

        loss = loss_label+loss_valid+loss_gen
        self.loss_total = loss
        self.learn = tf.train.AdamOptimizer().minimize(loss)

        self.initializer = tf.global_variables_initializer()

    def init_sess(self):

        graph = self.graph
        initializer = self.initializer

        sess = tf.Session(graph=graph)
        sess.run(initializer)

        self.sess = sess

    def run_sess(self):

        n_iter = self.n_batch
        get_batch = self.get_batch
        n_epoch = self.n_epoch

        sess = self.sess

        X = self.X 
        y_label = self.y_label
        y_valid = self.y_valid
        style = self.style

        acc_label = self.acc_label
        acc_valid = self.acc_valid
        acc_gen = self.acc_gen
        loss_total = self.loss_total
        learn = self.learn

        for i in range(n_iter):
            train_X, train_y_label, train_y_valid = get_batch(i)
            acc_label_, acc_valid_, acc_gen_, loss_total_, _ = sess.run(
                [acc_label, acc_valid, acc_gen, loss_total, learn],
                feed_dict={
                    X: train_X,
                    y_label: train_y_label,
                    y_valid: train_y_valid,
                    style: train_y_label
                }
            )
            sys.stdout.write(
                "\r% 5d | %5.2f%% | %8.7f | %8.7f | %8.7f | %8.7f"
                % (n_epoch,
                ((i+1)/n_iter*100),
                acc_label_,
                acc_valid_,
                acc_gen_,
                loss_total_)
            )
        print()