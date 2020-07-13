import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
dataset = np.array([1,2,3,4,5,6,7,8,9,10])
print(dataset.shape) # (10,)

data_gen = TimeseriesGenerator(dataset,dataset,5)
x_data = data_gen[0][0].reshape(-1,5,1)
y_data = data_gen[0][1].reshape(-1,1)
print(x_data,y_data)

output = 50
sequence_length = 5
input_dim = 1 


x = tf.placeholder(tf.float32,shape=[None,sequence_length,input_dim])
y = tf.placeholder(tf.int32, shape=[None, input_dim])

cell = tf.nn.rnn_cell.BasicLSTMCell(output)
hypothesis, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

weights = tf.ones([ 1, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=hypothesis, targets=y, weights=weights)

cost = tf.compat.v1.reduce_mean(tf.losses.mean_squared_error(y,hypothesis))

train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.compat.v1.argmax(hypothesis, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(sess.run(hypothesis, feed_dict={
    #       x: x_data, y: y_data}).shape)  # (5, 5, 50)
    for i in range(401):
        c,_ = sess.run([cost,train],feed_dict={x:x_data, y: y_data})
        print(i,c)
    p = sess.run(prediction,feed_dict={x:[10,11,12,13,14,15]})
    print(p)
