import tensorflow as tf

hello = tf.constant('Hello, Tensorflow!')
print(hello)

a = tf.constant(10)
b = tf.constant(20)
c = a + b
print(c)

sess = tf.Session()

print(sess.run(hello))
print(sess.run(c))

sess.close()


