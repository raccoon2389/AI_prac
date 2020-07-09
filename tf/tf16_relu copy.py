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
print(x_train.shape)
print(y_train[-1])

batch_size = 500
total_batch = int(len(x_train)/batch_size)
keep_prob = tf.compat.v1.placeholder(tf.float32)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, x_train.shape[1]])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, y_train.shape[1]])

layer_node = [512,512,512,256,10,10,10,10,10,10]

W1 = tf.get_variable("w1",[x_train.shape[1], layer_node[0]],initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.zeros([layer_node[0]]))

layer1 = tf.nn.relu(tf.matmul(x, W1)+b1)
layer11 = tf.nn.dropout(layer1,keep_prob=keep_prob)

W2 = tf.get_variable("w2", [layer_node[0],layer_node[1]],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.zeros([layer_node[1]]))
layer2 = tf.nn.relu(tf.matmul(layer11, W2)+b2)
layer22 = tf.nn.dropout(layer2, keep_prob=keep_prob)

W3 = tf.get_variable("w3", [layer_node[1], layer_node[2]],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.zeros([layer_node[2]]))
layer3 = tf.nn.relu(tf.matmul(layer22, W3)+b3)
layer33 = tf.nn.dropout(layer3, keep_prob=keep_prob)

W4 = tf.get_variable("w4", [layer_node[2], layer_node[3]],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.zeros([layer_node[3]]))
layer4 = tf.nn.relu(tf.matmul(layer33, W4)+b4)
layer44 = tf.nn.dropout(layer4, keep_prob=keep_prob)

W5 = tf.get_variable("w5", [layer_node[3], layer_node[4]],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.zeros([layer_node[4]]))
hypo = tf.nn.softmax(tf.matmul(layer44, W5)+b5)


# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=hypo))
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypo), axis=1))

predicted = tf.math.argmax(hypo,1)+1
acc = tf.reduce_mean(tf.cast(tf.equal(tf.math.argmax(hypo, 1), tf.math.argmax(y, 1)), dtype=tf.float32))

optimizer = tf.compat.v1.train.AdamOptimizer(
    learning_rate=0.005).minimize(loss)



with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for epoch in range(100):
        ave_cost = 0
        for i in range(total_batch):
            batch_xs,batch_ys = x_train[i*batch_size:(i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size]
        _, loss_val = sess.run([optimizer, loss], feed_dict={
                                  x: batch_xs, y: batch_ys,keep_prob:0.9})
        if epoch % 10 == 0:
            print(epoch, loss_val)
    a,l = sess.run([acc,loss],feed_dict={x:x_test,y:y_test,keep_prob:0.9})
    print("test : " , a,l)
