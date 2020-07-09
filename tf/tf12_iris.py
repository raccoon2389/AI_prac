import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris

x_data,y_data = load_iris(return_X_y=True)
y_data = tf.one_hot(y_data,3)
print(y_data)
print(x_data.shape,y_data.shape)
x = tf.placeholder(tf.float32,shape=[None,x_data.shape[1]])
y = tf.placeholder(tf.float32,shape=[None,y_data.shape[1]])

W = tf.Variable(tf.zeros([x_data.shape[1],y_data.shape[1]]))
b = tf.Variable(tf.zeros([y_data.shape[1]]))

hypo = tf.nn.softmax(tf.matmul(x,W)+b)

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypo),axis=1))

predicted = tf.arg_max(hypo,1)

acc = tf.reduce_mean(tf.cast(tf.equal(predicted, tf.argmax(y,1)), dtype=tf.float32))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_data = sess.run(y_data)
    for i in range(2001):
        _,loss_val,a = sess.run([optimizer,loss,acc],feed_dict={x:x_data,y : y_data})
        if i%200==0:
            print(i,loss_val,a)
