import numpy as np
import tensorflow as tf

x_data = np.array([[0,0],[0,1],[1,0],[1,1]])
y_data = np.array([0,1,1,0])
y_data = y_data.reshape(-1,1)

x = tf.placeholder(tf.float32, shape=[None, x_data.shape[1]])
y = tf.placeholder(tf.float32, shape=[None, y_data.shape[1]])

W = tf.Variable(tf.zeros([x_data.shape[1], y_data.shape[1]]))
b = tf.Variable(tf.zeros([y_data.shape[1]]))

hypo = tf.sigmoid(tf.matmul(x, W)+b)

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypo), axis=1))

predicted = tf.cast(hypo > 0.5, dtype=tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=0.01).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2001):
        _, loss_val, a = sess.run([optimizer, loss, acc], feed_dict={
                                  x: x_data, y: y_data})
        if i % 200 == 0:
            print(i, loss_val, a)
