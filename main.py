import numpy as np
import pandas as pd
import sys

import tensorflow as tf
from gensim.models import Word2Vec


class SSS():
    
    def __init__(self, sample_size):
        self._initialize(sample_size)
    
    def _initialize(self, sample_size):

        tf.set_random_seed(50554145)
        np.random.seed(50554145)
        
        df = pd.read_pickle('./data/df')
        df = df[df['label'] != 'neu']
        df = df[df['valid'] != 'fake']

        idxs = np.array(range(len(df)))
        np.random.shuffle(idxs)
        self.df = df.iloc[idxs][:sample_size]
        
        model = Word2Vec.load('./data/w2v_model')
        self.i2w = model.wv.index2word
        
        self.table = pd.read_pickle('./data/lookup_table')

    def set_hyper_parameters(self, batch_size):
        df = self.df
        self.batch_size = batch_size
        self.n_batch = len(df) // batch_size
    
    def get_batch(self, i):
    
        batch_size = self.batch_size
        df = self.df

        def get_X(sen):
            X = np.zeros(64, dtype=np.int32)+len(self.i2w)
            X[:len(sen)] = sen
            return X

        def get_y_label(label):
            idx = ['neg', 'pos'].index(label)
            y = np.array([0, 0])
            y[idx] = 1
            return y
               
        start, end = i*batch_size, (i+1)*batch_size
        df = df[start:end]

        Xs = np.array([get_X(sen) for sen in df['pos']])
        y_labels = np.array([get_y_label(label) for label in df['label']])

        return Xs, y_labels
    
    def check_sen(self, idx):
        
        X = self.X
        y_label = self.y_label
        gen = self.gen
        prob_l = self.prob_l
        prob_v = self.prob_v
        acc_l = self.acc_l
        acc_v = self.acc_v
        sess = self.sess

        df = self.df
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
                y_label: [label]
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
                y_label: [label_c]
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

        def get_output(sen):
            with tf.variable_scope('output', reuse=tf.AUTO_REUSE):
                lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(512)
                lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(512)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    lstm_cell_fw, lstm_cell_bw, sen, dtype=tf.float32
                )
                output = tf.concat(outputs, axis=1)
            return output

        def get_latent(output):
            with tf.variable_scope('latent', reuse=tf.AUTO_REUSE):
                conv = tf.reshape(tf.layers.conv1d(output, 512, 128),
                                  [-1, 512])
                conv_bn = tf.layers.batch_normalization(conv)
                conv_sg = tf.nn.sigmoid(conv_bn)
                latent = tf.layers.dropout(conv_sg, 0.5)
            return latent

        def sleep(latent, output, sen, y_label):
            with tf.variable_scope('sleep'):
                output_l = tf.concat([
                    tf.reshape(latent, shape=[-1, 1, 512]),
                    output
                ], axis=1)
                output_s = tf.concat([
                    output_l,
                    tf.transpose(
                        tf.zeros(shape=[129, tf.shape(y_label)[0], 2])+y_label,
                        [1, 0, 2]
                    )
                ], axis=2)
                conv = tf.layers.conv1d(output_s, 512, 66)
                conv_bn = tf.layers.batch_normalization(conv)
                conv_sg = tf.nn.sigmoid(conv_bn)
                latent = tf.layers.dropout(conv_sg, 0.5)
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=512)
                output_g, _ = tf.nn.dynamic_rnn(
                    lstm_cell, latent, dtype=tf.float32
                )
                w = tf.Variable(tf.truncated_normal([64, 64],
                                                    dtype=tf.float32))
                b = tf.Variable(tf.truncated_normal([64, 512],
                                                    dtype=tf.float32))
                gen = tf.map_fn(lambda x: tf.matmul(w, x)+b, output_g)
                loss_gen = tf.losses.mean_squared_error(sen, gen)
                self.loss_gen = loss_gen
                self.gen = gen
            return gen, loss_gen

        def wake(latent, y_label, y_valid):
            with tf.variable_scope('wake', reuse=tf.AUTO_REUSE):
                d0_l0 = tf.layers.dropout(
                    tf.layers.dense(latent, 256, activation=tf.sigmoid), 0.5
                )
                d0_l = tf.layers.dropout(
                    tf.layers.dense(d0_l0, 128, activation=tf.sigmoid), 0.5
                )
                d1_l = tf.layers.dropout(
                    tf.layers.dense(d0_l, 64, activation=tf.sigmoid), 0.5
                )
                d2_l = tf.layers.dense(d1_l, 2, activation=tf.nn.softmax)
                d0_v0 = tf.layers.dropout(
                    tf.layers.dense(latent, 256, activation=tf.sigmoid), 0.5
                )
                d0_v = tf.layers.dropout(
                    tf.layers.dense(d0_v0, 128, activation=tf.sigmoid), 0.5
                )
                d1_v = tf.layers.dropout(
                    tf.layers.dense(d0_v, 64, activation=tf.sigmoid), 0.5
                )
                d2_v = tf.layers.dense(d1_v, 2, activation=tf.nn.softmax)
                loss_l = tf.losses.softmax_cross_entropy(y_label, d2_l)        
                loss_v = tf.losses.softmax_cross_entropy(y_valid, d2_v)
                acc_l = tf.reduce_mean(tf.cast(
                    tf.equal(tf.argmax(d2_l, axis=1),
                             tf.argmax(y_label, axis=1)),
                    dtype=tf.float32
                ))
                acc_v = tf.reduce_mean(tf.cast(
                    tf.equal(tf.argmax(d2_v, axis=1),
                             tf.argmax(y_valid, axis=1)),
                    dtype=tf.float32
                ))
                self.prob_l = d2_l
                self.prob_v = d2_v
            return loss_l, loss_v, acc_l, acc_v

        def get_vs(scope_names):
            def get_vs(name):
                with tf.variable_scope(name) as scope:
                    vs = scope.global_variables()
                return vs
            vs = sum([get_vs(name) for name in scope_names], [])
            return vs

        tf.reset_default_graph()
        graph = tf.get_default_graph()
        self.graph = graph

        X = tf.placeholder(dtype=tf.int32, shape=[None, 64])
        y_label = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        self.X = X
        self.y_label = y_label

        y_valid = y_label*0+[0, 1]
        y_valid_s, y_label_s = -1*(y_valid-[1, 1]), -1*(y_label-[1, 1])
        y_valid_c = tf.concat([y_valid, y_valid_s], axis=0)
        y_label_c = tf.concat([y_label, y_label], axis=0)

        lookup_table = tf.Variable(table, dtype=tf.float32)
        X_ebd = tf.nn.embedding_lookup(lookup_table, X)
        self.lookup_table = lookup_table

        output = get_output(X_ebd)
        latent = get_latent(output)
        gen, loss_gen = sleep(
            latent, output, tf.stop_gradient(X_ebd), y_label
        )

        latent_s = get_latent(get_output(gen))
        latent_c = tf.concat([latent, latent_s], axis=0)
        loss_l, loss_v, acc_l, acc_v = wake(latent_c, y_label_c, y_valid_c)
        loss_l_s, loss_v_s, acc_l_s, acc_v_s = wake(latent_s,
                                                    y_label_s,
                                                    y_valid)

        vs4w = get_vs(['latent', 'output', 'wake'])
        vs4s = get_vs(['latent', 'output', 'sleep'])        
        loss_w = loss_l+loss_v*10
        loss_s = loss_l_s+loss_v_s*10+loss_gen
        self.acc_l, self.acc_v = acc_l, acc_v
        self.acc_l_s, self.acc_v_s = acc_l_s, acc_v_s
        self.learn_w = tf.train.AdamOptimizer().minimize(loss_w, var_list=vs4w)        
        self.learn_s = tf.train.AdamOptimizer().minimize(loss_s, var_list=vs4s)

        self.initializer = tf.global_variables_initializer()

    def init_sess(self):
        graph = self.graph        
        initializer = self.initializer
        sess = tf.Session(graph=graph)
        sess.run(initializer)
        tf.summary.FileWriter('./board').add_graph(graph)        
        self.sess = sess

    def run_sess(self):

        n_iter = self.n_batch
        get_batch = self.get_batch

        sess = self.sess

        X = self.X 
        y_label = self.y_label

        acc_l, acc_v = self.acc_l, self.acc_v
        acc_l_s, acc_v_s = self.acc_l_s, self.acc_v_s
        loss_gen = self.loss_gen

        learn_w = self.learn_w
        learn_s = self.learn_s

        lookup_table = self.lookup_table

        for i in range(n_iter):
            tr_X, tr_y_label = get_batch(i)
            acc_l_, acc_v_, acc_l_s_, acc_v_s_, loss_gen_, _, _ = sess.run(
                [acc_l, acc_v, acc_l_s, acc_v_s, loss_gen, learn_w, learn_s],
                feed_dict={X: tr_X, y_label: tr_y_label}
            )
            sys.stdout.write(
                ("\r%5.2f%% | acc-l %8.7f | acc-v %8.7f"
                 +" | acc-l-s %8.7f | acc-v-s %8.7f | loss-gen %8.7f")
                % (((i+1)/n_iter*100),
                   acc_l_,
                   acc_v_,
                   acc_l_s_,
                   acc_v_s_,
                   loss_gen_)
            )
        print()

        self.table = sess.run(lookup_table)
