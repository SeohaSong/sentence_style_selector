import numpy as np

import tensorflow as tf

from module_.utils.tf_agent import Tool


tool = Tool()
tool.set_hyper_parameters(
    batch_size=128
)
n_iter = tool.n_batch
table = tool.table


tf.set_random_seed(0)
np.random.seed(0)

tf.reset_default_graph()
graph = tf.get_default_graph()

X = tf.placeholder(dtype=tf.int32, shape=[None, 64])
y = tf.placeholder(dtype=tf.float32, shape=[None, 2])
y_valid = tf.placeholder(dtype=tf.float32, shape=[None, 2])

lookup_table = tf.Variable(table, dtype=tf.float32)
X_ebd = tf.nn.embedding_lookup(lookup_table, X)

lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(128)
lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(128)
output, state = tf.nn.bidirectional_dynamic_rnn(
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

acc = tf.reduce_mean(tf.cast(
    tf.equal(tf.argmax(fin, axis=1), tf.argmax(y, axis=1)),
    dtype=tf.float32
))
loss = tf.losses.softmax_cross_entropy(y, fin)
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
output_gen, state_gen = tf.nn.dynamic_rnn(
    lstm_cell, logit, dtype=tf.float32
)

w_gen = tf.Variable(tf.truncated_normal([128, 256], dtype=tf.float32))
b_gen = tf.Variable(tf.truncated_normal([128, 512], dtype=tf.float32))
fin_gen = tf.map_fn(lambda x: tf.matmul(w_gen, x)+b_gen, output_gen)

loss_gen = tf.losses.mean_squared_error(X_ebd, fin_gen)
############## for gen ##########

loss = loss_label+loss_valid+loss_gen
learn = tf.train.AdamOptimizer().minimize(loss)

initilizer = tf.global_variables_initializer()

sess = tf.Session(graph=graph)
sess.run(initilizer)