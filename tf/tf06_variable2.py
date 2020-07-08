
import tensorflow as tf
tf.set_random_seed(777)

x = [1,2,3]
W = tf.Variable([0.3],dtype=tf.float32, name='weight')
b = tf.Variable([1],dtype=tf.float32,name='bias')
print(b)
H= W*x + b 

sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(H)
print(aaa)
sess.close()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = H.eval()
print(bbb)
sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = H.eval(session=sess)
print(ccc)
sess.close