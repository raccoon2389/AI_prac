
import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(777)

x_data = [[75., 51., 65.], [92, 98, 11], [
    89, 31, 33], [99, 33, 100], [17, 66, 79]]
y_data = [[152], [185], [180], [205], [142]]

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_normal([3, 1], name='weight1'))
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x, W) + b

cost = tf.reduce_mean(tf.square(hypothesis-y))
# cost = tf.losses.mean_squared_error(hypothesis, y_train)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00005)
train = optimizer.minimize(cost)
# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
# train = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        curr_cost_, cost_val, _ = sess.run(
            [cost, hypothesis, train], feed_dict={x: x_data, y: y_data})
        if step % 20 == 0:
            print(step, cost_val, curr_cost_)
    sess.close()
