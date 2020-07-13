from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import numpy as np

# data = hihello
#1. DATA
idx2char = ['e', 'h', 'i', 'l', 'o']

_data = np.array([['h', 'i', 'h', 'e', 'l', 'l', 'o']]).reshape(7, 1)
print(_data.shape)  # (1, 7)
print(_data)        # [['h' 'i' 'h' 'e' 'l' 'l' 'o']]
print(type(_data))  # <class 'numpy.ndarray'>

# ENC = tf.one_hot(_data, depth=5, axis=1, dtype=tf.float64)
# print(ENC)

enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray()

print(_data)
print(type(_data))
print(_data.dtype)

x_data = _data[:6, ]
y_data = _data[1:, ]

# print('=========================')
# print(X_DATA)
# print('=========================')
# print(Y_DATA)

y_data = np.argmax(y_data, axis=1)
# print('=========================')
# print(Y_DATA)

x_data = x_data.reshape(1, 6, 5)
y_data = y_data.reshape(1, 6)
# print(Y_DATA.shape)

X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 6, 5])
Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 6])

SEQUENCE_LENGTH = 6
INPUT_DIM = 5
OUTPUT = 5
BATCH_SIZE = 1      # All row

X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[
                             None, SEQUENCE_LENGTH, INPUT_DIM])
Y = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, SEQUENCE_LENGTH])

print(X)
print(Y)
print(Y.dtype)

#. MODEL

# cell = tf.nn.rnn_cell.BasicLSTMCell(OUTPUT)
cell = tf.keras.layers.LSTMCell(OUTPUT)
hypothesis, _state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
print(hypothesis)   # (?, 6, 100)


# COMPILE
weights = tf.ones([1, SEQUENCE_LENGTH])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=hypothesis, targets=Y, weights=weights)

cost = tf.compat.v1.reduce_mean(sequence_loss)

train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.compat.v1.argmax(hypothesis, axis=2)

# TRAINING
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(401):
        loss, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})
        print(i, "LOSS : ", loss, "PREDICTION : ", result)

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\nPREDICTION STR : ", ''.join(result_str))
