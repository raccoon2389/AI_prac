
import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(777)

x = [1.,2.,3.]
y = [3.,5.,7.]


W= tf.placeholder(tf.float32)
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x * W

cost = tf.reduce_mean(tf.square(hypothesis-y))
# cost = tf.losses.mean_squared_error(hypothesis, y_train)

# train = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)
# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
# train = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
w_history =[]
cost_history = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(-30,50):
        curr_w = i*0.1
        curr_cost = sess.run(cost,feed_dict={W :curr_w})
        w_history.append(curr_w)
        cost_history.append(curr_cost)
    
    plt.plot(w_history,curr_cost)
    plt.show()
        

    sess.close()
