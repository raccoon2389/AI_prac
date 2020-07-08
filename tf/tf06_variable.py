
import tensorflow as tf
tf.set_random_seed(777)
W = tf.Variable([0.3], name='weight')
W = W*2
sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(W)
print(aaa)
sess.close()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = W.eval()
print(bbb)
sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = W.eval(session=sess)
print(ccc)
sess.close