import tensorflow as tf

node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1,node2)

sess = tf.Session()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a+b

print(sess.run(adder_node, feed_dict={a: 3, b:4.5}))
print(sess.run(adder_node, feed_dict={a: [1,3], b:[2,4.5]}))

add_and_triple = adder_node*3
print(sess.run(add_and_triple, feed_dict={a: [1, 3], b: [2, 4.5]}))
#placeholder 는 input과 비슷한 개념이다
#들어가는 노드의 위치를 나타낸가도 봐야할듯

