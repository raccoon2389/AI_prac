import numpy as np
import tensorflow as tf

idx2char = ['e','h','i','l','o']

_data = np.array([['h', 'i', 'h', 'e', 'l', 'l', 'o']], dtype=np.str).reshape(-1,1)
# print(_data.shape)  # (7,1)
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray()

print('==============================')
# print(_data)    # [[0. 1. 0. 0. 0.]
                # [0. 0. 1. 0. 0.]
                # [0. 1. 0. 0. 0.]
                # [1. 0. 0. 0. 0.]
                # [0. 0. 0. 1. 0.]
                # [0. 0. 0. 1. 0.]
                # [0. 0. 0. 0. 1.]]
# print(_data.shape)  # (7, 5)
x_data= _data[:6,].reshape(1,6,5)
y_data= _data[1:,]

# print(x)
# print(y)

y_data = np.argmax(y_data, axis=1).reshape(1, 6)
# print(y_data)  # [2 1 0 3 3 4]


sequence_length = 6
input_dim = 5
output = 100
x = tf.compat.v1.placeholder(tf.float32, (None, sequence_length, input_dim))
y = tf.compat.v1.placeholder(tf.int32, (None,sequence_length))

print(x)    # Tensor("Placeholder:0", shape=(?, 6, 5), dtype=float32)
print(y)    # Tensor("Placeholder_1:0", shape=(?, 6), dtype=float32)

cell = tf.nn.rnn_cell.BasicLSTMCell(output)
hypothesis, _states = tf.nn.dynamic_rnn(cell, x,dtype=tf.float32)

weights = tf.ones([1, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=hypothesis, targets=y, weights=weights)

cost = tf.compat.v1.reduce_mean(sequence_loss)

train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.compat.v1.argmax(hypothesis, axis=2)
print('good')

print(hypothesis)

prediction = tf.argmax(hypothesis,axis=2)
# enc.inverse_transform(y_data)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(hypothesis, feed_dict={x: x_data, y: y_data}).shape)
    # for i in range(401):
    #     cost,_ = sess.run([cost,train],feed_dict={x:x_data,y:y_data})
    #     result = sess.run(prediction, feed_dict={x: x_data, y: y_data})
    #     result = [idx2char[c]for c in np.squeeze(result)]
    #     print('cost : ',cost)
    #     print("\nPred : ",''.k)
