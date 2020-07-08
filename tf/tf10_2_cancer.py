from sklearn.datasets import load_breast_cancer
import tensorflow as tf


x_data, y_data = load_breast_cancer(return_X_y=True)
# y_data = y_data.reshape(-1, 1)
print(x_data.shape, y_data.shape)
x = tf.placeholder(tf.float32, shape=(None, x_data.shape[1]))
y = tf.placeholder(tf.float32, shape=(None, 1))

# print(x_data.shape)
# print(y_data)
W = tf.Variable(tf.zeros([x_data.shape[1], 1]))
b = tf.Variable(tf.zeros([y_data.shape[1]]))

hypo = tf.sigmoid(tf.matmul(x,W) + b)

cost = -tf.reduce_mean(y*tf.log(hypo)+(1-y)*tf.log(1-hypo))

train = tf.train.GradientDescentOptimizer(0.0000003).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(8001):
        c, _ = sess.run([cost, train], feed_dict={x: x_data, y: y_data})
        if i % 20 == 0:
            print(i, c)
