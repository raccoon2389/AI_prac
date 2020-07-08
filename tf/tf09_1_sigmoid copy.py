
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
tf.set_random_seed(777)

x_data = [[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8]]
y_data=[[0],[0],[0],[1],[1],[1]]

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_normal([3,1], name='weight1'))
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x,W) + b)

# cost = tf.reduce_mean(tf.square(hypothesis-y))
predicted = tf.cast(hypothesis>0.5, dtype= tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(predicted,y),dtype=tf.float32))
# cost = tf.losses.mean_squared_error(hypothesis, y_train)

cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00005)
train = optimizer.minimize(cost)

# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
# train = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        curr_cost_,_ = sess.run([cost,train], feed_dict={x: x_data,y:y_data})
        if step % 20 == 0:
            print(step,curr_cost_)
    
    h,c,a = sess.run([hypothesis,predicted,acc],feed_dict={x:x_data,y:y_data})
    print(f"hypo : {h}\ncost : {c}\nacc : {a}")
    sess.close()
