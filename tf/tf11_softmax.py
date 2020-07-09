import tensorflow as tf
import numpy as np

x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5],
          [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 6, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0],
          [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

x = tf.placeholder(dtype=tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])

W = tf.Variable(tf.zeros([4, 3]))
b = tf.Variable(tf.zeros([3]))

hypo = tf.nn.softmax(tf.matmul(x, W)+b)
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypo), axis=1))
optimizer = tf.train.AdamOptimizer(
    learning_rate=0.01).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2001):
        _, cost_val = sess.run([optimizer, loss], feed_dict={
                               x: x_data, y: y_data})
        if i%200 == 0:
            print(i,cost_val)
    a = sess.run(hypo , feed_dict={x:[[1,11,7,9]]})
    print(sess.run(tf.arg_max(a,1)))
    b = sess.run(hypo, feed_dict={x: [[1,3,4,3]]})
    print(sess.run(tf.arg_max(b, 1)))
    c = sess.run(hypo, feed_dict={x: [[11,33,4,13]]})
    print(sess.run(tf.arg_max(c, 1)))
    print(type(c))
    alll = sess.run(hypo, feed_dict={
                    x: [np.append(a, 0), np.append(b, 0), np.append(c, 0)]})
    print(sess.run(tf.arg_max(alll, 1)))
