
import tensorflow as tf
tf.set_random_seed(777)

x_train = tf.placeholder(tf.float32)
y_train = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x_train * W + b
x_test = tf.placeholder(tf.float32)

# cost = tf.reduce_mean(tf.square(hypothesis-y_train))
cost = tf.losses.mean_squared_error(hypothesis,y_train)

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b],feed_dict={x_train:[1,2,3,4,5],y_train:[3,5,7,9,11]})

        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)
    print(sess.run(hypothesis, feed_dict={x_train: [5, 6]}))
    print(sess.run(hypothesis, feed_dict={x_train: [6,7,8]}))

    sess.close()

