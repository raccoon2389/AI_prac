import numpy as np
import tensorflow as tf
from keras.datasets import mnist


(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train, x_test = x_train.reshape(-1, 28*28), x_test.reshape(-1, 28*28)
with tf.Session() as sess:
    y_train = sess.run(tf.one_hot(y_train, 10))
    y_test = sess.run(tf.one_hot(y_test, 10))

print(y_train)
x = tf.compat.v1.placeholder(tf.float32, shape=[None, x_train.shape[1]])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, y_train.shape[1]])

layer_node = [1,2,3,4,5,6,7,8,9,10]

W1 = tf.Variable(tf.zeros([x_train.shape[1], layer_node[0]]))
b1 = tf.Variable(tf.zeros([layer_node[0]]))

layer1 = tf.matmul(x, W1)+b1

W2 = tf.Variable(tf.zeros([layer_node[0], layer_node[1]]))
b2 = tf.Variable(tf.zeros([layer_node[1]]))

layer2 = tf.matmul(layer1, W2)+b2

W3 = tf.Variable(tf.zeros([layer_node[1], layer_node[2]]))
b3 = tf.Variable(tf.zeros([layer_node[2]]))

layer3 = tf.matmul(layer2, W3)+b3
W4 = tf.Variable(tf.zeros([layer_node[2], layer_node[3]]))
b4 = tf.Variable(tf.zeros([layer_node[3]]))

layer4 = tf.matmul(layer3, W4)+b4

W5 = tf.Variable(tf.zeros([layer_node[3], layer_node[4]]))
b5 = tf.Variable(tf.zeros([layer_node[4]]))

layer5 = tf.matmul(layer4, W5)+b5

W6 = tf.Variable(tf.zeros([layer_node[4], layer_node[5]]))
b6 = tf.Variable(tf.zeros([layer_node[5]]))

layer6 = tf.matmul(layer5, W6)+b6

W7 = tf.Variable(tf.zeros([layer_node[5], layer_node[6]]))
b7 = tf.Variable(tf.zeros([layer_node[6]]))

layer7 = tf.matmul(layer6, W7)+b7

W8 = tf.Variable(tf.zeros([layer_node[6], layer_node[7]]))
b8 = tf.Variable(tf.zeros([layer_node[7]]))

layer8 = tf.matmul(layer7, W8)+b8


W9 = tf.Variable(tf.zeros([layer_node[7], layer_node[8]]))
b9 = tf.Variable(tf.zeros([layer_node[8]]))

layer9 = tf.matmul(layer8, W9)+b9


W10 = tf.Variable(tf.zeros([layer_node[8], layer_node[9]]))
b10 = tf.Variable(tf.zeros([layer_node[9]]))

hypo = tf.nn.softmax(tf.matmul(layer9, W10)+b10)

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypo), axis=1))

predicted = tf.math.argmax(hypo,1)+1
acc = tf.reduce_mean(tf.cast(tf.equal(tf.math.argmax(hypo, 1), tf.math.argmax(y, 1)), dtype=tf.float32))

optimizer = tf.compat.v1.train.AdamOptimizer(
    learning_rate=0.1).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(2001):
        _, loss_val, a,h = sess.run([optimizer, loss, acc,hypo], feed_dict={
                                  x: x_train, y: y_train})
        if i % 200 == 0:
            print(i, h,loss_val, a)
    a,l = sess.run([acc,loss],feed_dict={x:x_test,y:y_test})
    print("test : " , a,l)
