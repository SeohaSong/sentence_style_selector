import numpy as np
import pandas as pd
import sys

from gensim.models import Word2Vec
import tensorflow as tf

class TF_Agent():
    
    def __init__(self):
        self._initialize()
    
    def _initialize(self):

        tf.set_random_seed(50554145)
        np.random.seed(50554145)
        
        df = pd.read_pickle('./data/df')
        df = df[df['label'] != 'neu']

        df_real = df[df['valid'] == 'real']
        df_fake = df[df['valid'] == 'fake']

        df_init = pd.concat([df_real, df_fake], axis=0)
        df_sleep = pd.concat([df_real, df_real], axis=0)
        df_wake = df_sleep.copy()
        df_wake['valid'] = df_init['valid']

        idxs = np.array(range(len(df)))
        np.random.shuffle(idxs)
        self.df_init = df_init.iloc[idxs]
        self.df_wake = df_wake.iloc[idxs]
        self.df_sleep = df_sleep.iloc[idxs]
        
        model = Word2Vec.load('./data/w2v_model')
        self.i2w = model.wv.index2word
        
        self.table = pd.read_pickle('./data/lookup_table')

        tf.reset_default_graph()
        self.graph = tf.get_default_graph()

    def set_hyper_parameters(self, batch_size):

        df_wake = self.df_wake
        
        self.batch_size = batch_size
        self.n_batch = len(df_wake) // batch_size
        self.n_epoch = 0
    
    def get_batch(self, i, case):
    
        batch_size = self.batch_size
        sess = self.sess
        X = self.X 
        X_ebd_f = self.X_ebd_f
        y_label = self.y_label
        gen = self.gen
        if case == 'init':
            df = self.df_init
        elif case == 'wake':
            df = self.df_wake
        else:
            df = self.df_sleep

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

        if case == 'wake':
            tr_X = Xs[y_valids[:, 0] == 1]
            tr_y_labels = (y_labels[y_valids[:, 0] == 1]-np.array([1, 1]))*-1
            X_ebd_fs = sess.run(
                gen,
                feed_dict={
                    X: tr_X,
                    X_ebd_f: np.ones([0, 64, 512]),
                    y_label: tr_y_labels
                }
            )
            Xs = Xs[y_valids[:, 0] == 0]
            y_labels = y_labels[y_valids[:, 0] == 0]
            y_valids = y_valids[y_valids[:, 0] == 0]
            y_labels = np.concatenate([y_labels, tr_y_labels])
            y_valids = np.concatenate([y_valids, np.array([[1, 0]]*len(tr_X))])
        else:
            X_ebd_fs = np.ones([0, 64, 512])
        if case == 'sleep':
            y_labels = (y_labels-np.array([1, 1]))*-1

        return Xs, X_ebd_fs, y_labels, y_valids
    
    def check_sen(self, idx):
        
        X = self.X
        X_ebd_f = self.X_ebd_f
        y_label = self.y_label
        y_valid = self.y_valid
        gen = self.gen
        prob_l = self.prob_l
        prob_v = self.prob_v
        acc_l = self.acc_l
        acc_v = self.acc_v
        sess = self.sess

        df = self.df_sleep
        i2w = self.i2w
        table = self.table

        one = df.iloc[idx]

        lst = np.zeros(64, dtype=np.int32)-1
        lst[:len(one['pos'])] = one['pos']

        label = np.array([0, 0])
        label[['neg', 'pos'].index(one['label'])] = 1

        prob_l_r, prob_v_r, acc_l_r, acc_v_r, [mat] = sess.run(
            [prob_l, prob_v, acc_l, acc_v, gen], feed_dict={
                X: [lst],
                X_ebd_f: np.ones([0, 64, 512]),
                y_label: [label],
                y_valid: [[0, 1]]
            }
        )

        def convert_lst2sen(lst):
            sen = ' '.join([i2w[i].split("-")[0] for i in lst if i > 0])
            return sen

        def get_idx(i, v):
            idx = np.linalg.norm(table-v, axis=1).argmin()
            sys.stdout.write("\r%5.2f%%" % ((i+1)/len(mat)*100))
            return idx
        
        def get_word(idx):
            word = i2w[idx]
            word = word.split("-")[0]
            return word

        idxs = np.array([get_idx(i, v) for i, v in enumerate(mat)])
        print()

        label_c = (label-np.array([1, 1]))*-1
        prob_l_f, prob_v_f, acc_l_f, acc_v_f = sess.run(
            [prob_l, prob_v, acc_l, acc_v], feed_dict={
                X: [idxs],
                X_ebd_f: np.ones([0, 64, 512]),
                y_label: [label_c],
                y_valid: [[1, 0]]
            }
        )
        sen_real = convert_lst2sen(lst)
        sen_gen = ' '.join([get_word(i) for i in idxs if i != len(i2w)])

        loss_dic = {
            "real": {
                'label': (label, prob_l_r, acc_l_r),
                'valid': (np.array([0, 1]), prob_v_r, acc_v_r)
            },
            "fake": {
                'label': (label_c, prob_l_f, acc_l_f),
                'valid': (np.array([1, 0]), prob_v_f, acc_v_f)
            }
        }

        return sen_real, sen_gen, loss_dic

    def init_graph(self):
        
        table = self.table

        def wake(sen, reuse, y_label, y_valid):
            with tf.variable_scope('wake', reuse=reuse):
                lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(256)
                lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(256)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    lstm_cell_fw, lstm_cell_bw, sen, dtype=tf.float32
                )
                output = tf.concat(outputs, axis=1)
                conv = tf.reshape(tf.layers.conv1d(output, 256, 128),
                                  [-1, 256])
                conv_bn = tf.layers.batch_normalization(conv)
                conv_sg = tf.nn.sigmoid(conv_bn)
                latent = tf.layers.dropout(conv_sg, 0.2)
                d0_l = tf.layers.dropout(
                    tf.layers.dense(latent, 128, activation=tf.nn.relu), 0.2
                )
                d1_l = tf.layers.dropout(
                    tf.layers.dense(d0_l, 64, activation=tf.nn.relu), 0.2
                )
                d2_l = tf.layers.dense(d1_l, 2, activation=tf.nn.softmax)
                d0_v = tf.layers.dropout(
                    tf.layers.dense(latent, 128, activation=tf.nn.relu), 0.2
                )
                d1_v = tf.layers.dropout(
                    tf.layers.dense(d0_v, 64, activation=tf.nn.relu), 0.2
                )
                d2_v = tf.layers.dense(d1_v, 2, activation=tf.nn.softmax)
                loss_l = tf.losses.softmax_cross_entropy(y_label, d2_l)        
                loss_v = tf.losses.softmax_cross_entropy(y_valid, d2_v)
                self.acc_l = tf.reduce_mean(tf.cast(
                    tf.equal(tf.argmax(d2_l, axis=1),
                             tf.argmax(y_label, axis=1)),
                    dtype=tf.float32
                ))
                self.acc_v = tf.reduce_mean(tf.cast(
                    tf.equal(tf.argmax(d2_v, axis=1),
                             tf.argmax(y_valid, axis=1)),
                    dtype=tf.float32
                ))
                self.prob_l = d2_l
                self.prob_v = d2_v
            if reuse == False:
                return loss_l, loss_v, output, latent
            else:
                return loss_l, loss_v

        def sleep(output, latent, y_label, X_ebd):
            with tf.variable_scope('sleep') as scope:
                output_l = tf.concat([
                    tf.reshape(latent, shape=[-1, 1, 256]),
                    output
                ], axis=1)
                output_s = tf.concat([
                    output_l,
                    tf.transpose(
                        tf.zeros(shape=[129, tf.shape(y_label)[0], 2])+y_label,
                        [1, 0, 2]
                    )
                ], axis=2)
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=512)
                output_g, _ = tf.nn.dynamic_rnn(
                    lstm_cell, output_s, dtype=tf.float32
                )
                w = tf.Variable(tf.truncated_normal([64, 129],
                                                    dtype=tf.float32))
                b = tf.Variable(tf.truncated_normal([64, 512],
                                                    dtype=tf.float32))
                gen = tf.map_fn(lambda x: tf.matmul(w, x)+b, output_g)
                loss_gen = tf.losses.mean_squared_error(X_ebd, gen)
                sleep_vars = scope.global_variables()
                self.loss_gen = loss_gen
                self.gen = gen
            return gen, loss_gen, sleep_vars

        X = tf.placeholder(dtype=tf.int32, shape=[None, 64])
        X_ebd_f = tf.placeholder(dtype=tf.float32, shape=[None, 64, 512])
        y_label = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        y_valid = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        self.X = X
        self.X_ebd_f = X_ebd_f
        self.y_label = y_label
        self.y_valid = y_valid
        
        lookup_table = tf.Variable(table, dtype=tf.float32)
        X_ebd_r = tf.nn.embedding_lookup(lookup_table, X)
        self.lookup_table = lookup_table
        X_ebd = tf.concat([X_ebd_r, X_ebd_f], axis=0)

        loss_l, loss_v, output, latent = wake(X_ebd, False, y_label, y_valid)
        gen, loss_gen, sleep_vars = sleep(output, latent, y_label, X_ebd)
        loss_i = loss_l+loss_v+loss_gen
        self.loss_total_i = loss_i
        self.learn_i = tf.train.AdamOptimizer().minimize(loss_i)
        loss_w = loss_l+loss_v
        self.loss_total_w = loss_w
        self.learn_w = tf.train.AdamOptimizer().minimize(loss_w)
        
        loss_l_s, loss_v_s = wake(gen, True, y_label, y_valid)
        loss_s = loss_l_s+loss_v_s+loss_gen*3
        self.loss_total_s = loss_s
        self.learn_s = tf.train.AdamOptimizer().minimize(
            loss_s, var_list=sleep_vars
        )

        self.initializer = tf.global_variables_initializer()

    def init_sess(self):
        graph = self.graph
        initializer = self.initializer
        sess = tf.Session(graph=graph)
        sess.run(initializer)
        self.sess = sess

    def run_sess_wake(self, case='wake'):

        n_iter = self.n_batch
        get_batch = self.get_batch

        sess = self.sess

        X = self.X 
        X_ebd_f = self.X_ebd_f
        y_label = self.y_label
        y_valid = self.y_valid

        acc_l = self.acc_l
        acc_v = self.acc_v
        loss_gen = self.loss_gen
        if case == 'init':
            loss_total_w = self.loss_total_i
            learn_w = self.learn_i
        elif case == 'wake':
            loss_total_w = self.loss_total_w
            learn_w = self.learn_w

        lookup_table = self.lookup_table

        for i in range(n_iter):
            tr_X, tr_X_ebd_f, tr_y_label, tr_y_valid = get_batch(i, case)
            acc_l_, acc_v_, loss_gen_, loss_total_w_, _ = sess.run(
                [acc_l, acc_v, loss_gen, loss_total_w, learn_w],
                feed_dict={
                    X: tr_X,
                    X_ebd_f: tr_X_ebd_f,
                    y_label: tr_y_label,
                    y_valid: tr_y_valid
                }
            )
            sys.stdout.write(
                ("\r%5.2f%% | acc-l %8.7f | "
                 +"acc-v %8.7f | loss-gen %8.7f | total-loss-w %8.7f")
                % (((i+1)/n_iter*100),
                   acc_l_,
                   acc_v_,
                   loss_gen_,
                   loss_total_w_)
            )
        print()

        self.table = sess.run(lookup_table)
        if case == 'init':
            self.run_sess_sleep()

    def run_sess_sleep(self):

        n_iter = self.n_batch
        get_batch = self.get_batch

        sess = self.sess

        X = self.X 
        X_ebd_f = self.X_ebd_f
        y_label = self.y_label
        y_valid = self.y_valid

        acc_l = self.acc_l
        acc_v = self.acc_v
        loss_gen = self.loss_gen
        loss_total_s = self.loss_total_s
        learn_s = self.learn_s

        for i in range(n_iter):
            tr_X, tr_X_ebd_f, tr_y_label, tr_y_valid = get_batch(i, 'sleep')
            acc_l_, acc_v_, loss_gen_, loss_total_s_, _ = sess.run(
                [acc_l, acc_v, loss_gen, loss_total_s, learn_s],
                feed_dict={
                    X: tr_X,
                    X_ebd_f: tr_X_ebd_f,
                    y_label: tr_y_label,
                    y_valid: tr_y_valid
                }
            )
            sys.stdout.write(
                ("\r%5.2f%% | acc-l %8.7f | "
                 +"acc-v %8.7f | loss-gen %8.7f | total-loss-s %8.7f")
                % (((i+1)/n_iter*100),
                   acc_l_,
                   acc_v_,
                   loss_gen_,
                   loss_total_s_)
            )
        print()
