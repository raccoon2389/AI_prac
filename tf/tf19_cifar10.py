import numpy as np
import tensorflow as tf
from keras.datasets import cifar10


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train.reshape(-1, x_train.shape[1], x_train.shape[2], 1) / \
    255., x_test.reshape(-1, x_train.shape[1], x_train.shape[2], 1)/255.
print(y_train[-1])
with tf.Session() as sess:
    y_train = sess.run(tf.one_hot(y_train, 10))
    y_test = sess.run(tf.one_hot(y_test, 10))
    sess.close()
print(x_train.shape)
print(y_train[-1])

batch_size = 500
total_batch = int(len(x_train)/batch_size)
rate = tf.compat.v1.placeholder(tf.float32)

x = tf.compat.v1.placeholder(
    tf.float32, shape=[None, x_train.shape[1], x_train.shape[2], 1])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, y_train.shape[1]])

layer_node = [512, 512, 512, 256, 10, 10, 10, 10, 10, 10]

W1 = tf.get_variable("w1", shape=[3, 3, 1, 32],  # [커널사이즈,컬러,아웃풋]
                     initializer=tf.contrib.layers.xavier_initializer())

layer1 = tf.nn.relu(tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME'))
layer11 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[
                         1, 1, 1, 1], padding='SAME')


W2 = tf.get_variable("w2", shape=[3, 3, 1, 32],
                     initializer=tf.contrib.layers.xavier_initializer())
layer2 = tf.nn.relu(tf.nn.conv2d(
    layer11, W2, strides=[1, 1, 1, 1], padding='SAME'))
layer22 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[
                         1, 1, 1, 1], padding='SAME')
layer222 = tf.reshape(layer22, [-1, 7*7*64])

W3 = tf.get_variable("w3", [7*7*64, layer_node[2]],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.zeros([layer_node[2]]))
layer3 = tf.nn.relu(tf.matmul(layer222, W3)+b3)
layer33 = tf.nn.dropout(layer3, rate=rate)

W4 = tf.get_variable("w4", [layer_node[2], 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.zeros(10))
layer4 = tf.nn.relu(tf.matmul(layer33, W4)+b4)
hypo = tf.nn.softmax(layer4)


# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=hypo))
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypo), axis=1))

predicted = tf.math.argmax(hypo, 1)+1
acc = tf.reduce_mean(tf.cast(tf.equal(tf.math.argmax(
    hypo, 1), tf.math.argmax(y, 1)), dtype=tf.float32))

optimizer = tf.compat.v1.train.AdamOptimizer(
    learning_rate=0.005).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for epoch in range(100):
        avg_cost = 0
        for i in range(total_batch):
            batch_xs, batch_ys = x_train[i*batch_size:(
                i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size]
            _, loss_val = sess.run([optimizer, loss], feed_dict={
                x: batch_xs, y: batch_ys, rate: 0.1})
            avg_cost += loss_val/batch_size
            # avg_cost += loss_val/total_batch
        if epoch % 1 == 0:
            print(epoch, avg_cost)
    a, l = sess.run([acc, loss], feed_dict={x: x_test, y: y_test, rate: 0.1})
    print("test : ", a, l)
