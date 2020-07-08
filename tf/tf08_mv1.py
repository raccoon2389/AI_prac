
import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(777)

x1_data = [73.,93.,78.,96.,73.]
x2_data = [80.,88.,91.,98.,66.]
x3_data = [75.,93.,90.,100.,70.]

y_data=[152.,185.,180.,196.,142]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
W1 = tf.Variable(tf.random_normal([1], name='weight1'))
W2 = tf.Variable(tf.random_normal([1], name='weight2'))
W3 = tf.Variable(tf.random_normal([1], name='weight3'))
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x1*W1+x2*W2+x3*W3+b

cost = tf.reduce_mean(tf.square(hypothesis-y))
# cost = tf.losses.mean_squared_error(hypothesis, y_train)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00000005)
train = optimizer.minimize(cost)
# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
# train = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        curr_cost_, cost_val, _ = sess.run([cost,hypothesis,train], feed_dict={x1: x1_data, x2: x2_data, x3: x3_data,y:y_data})
        if step % 20 == 0:
            print(step, cost_val,curr_cost_)
    sess.close()
