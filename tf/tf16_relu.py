import numpy as np
import tensorflow as tf
from keras.datasets import mnist


(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train, x_test = x_train.reshape(-1, 28*28)/255., x_test.reshape(-1, 28*28)/255.
print(y_train[-1])
with tf.Session() as sess:
    y_train = sess.run(tf.one_hot(y_train, 10))
    y_test = sess.run(tf.one_hot(y_test, 10))
    sess.close()
print(y_train.shape)
print(y_train[-1])
x = tf.compat.v1.placeholder(tf.float32, shape=[None, x_train.shape[1]])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, y_train.shape[1]])

layer_node = [10,10,10,10,10,10,10,10,10,10]

W1 = tf.Variable(tf.random.normal([x_train.shape[1], layer_node[0]]))
b1 = tf.Variable(tf.zeros([layer_node[0]]))
layer1 = tf.nn.relu(tf.matmul(x, W1)+b1)
W2 = tf.Variable(tf.zeros([layer_node[0], layer_node[1]]))
b2 = tf.Variable(tf.zeros([layer_node[1]]))


hypo = tf.nn.softmax(tf.matmul(layer1, W2)+b2)


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=hypo))
# loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypo), axis=1))

predicted = tf.math.argmax(hypo,1)+1
acc = tf.reduce_mean(tf.cast(tf.equal(tf.math.argmax(hypo, 1), tf.math.argmax(y, 1)), dtype=tf.float32))

optimizer = tf.compat.v1.train.AdamOptimizer(
    learning_rate=0.05).minimize(loss)

batch_size= 100
total_batch = int(len(x_train/batch_size))
keep_prob = tf.compat.v1.placeholder(tf.float32)

with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(100):
        _, loss_val, a,h = sess.run([optimizer, loss, acc,hypo], feed_dict={
                                  x: x_train, y: y_train})
        if i % 10 == 0:
            print(i, loss_val, a)
    a,l = sess.run([acc,loss],feed_dict={x:x_test,y:y_test})
    print("test : " , a,l)
