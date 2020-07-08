from sklearn.datasets import load_diabetes
import tensorflow as tf


x_data, y_data = load_diabetes(return_X_y=True)
y_data = y_data.reshape(-1, 1)
print(x_data.shape, y_data.shape)
x = tf.placeholder(tf.float32, shape=(None, x_data.shape[1]))
y = tf.placeholder(tf.float32, shape=(None, y_data.shape[1]))

# print(x_data.shape)
# print(y_data)
W = tf.Variable(tf.random_normal([10,1]))
b = tf.Variable(tf.random_normal([1]))

hypo = tf.matmul(x,W) + b

cost = tf.reduce_mean(tf.losses.mean_squared_error(y_data, hypo))

train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2001):
        c, _ = sess.run([cost, train], feed_dict={x: x_data, y: y_data})
        if i % 20 == 0:
            print(i, c)
