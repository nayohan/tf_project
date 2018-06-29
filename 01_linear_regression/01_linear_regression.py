import tensorflow as tf

X = tf.placeholder(tf.float32, [None,3])
x_data = [[1,2,3],[5,6,7]]

W = tf.Variable(tf.random_normal([3,2]), name="W")
b = tf.Variable(tf.random_normal([2,1]), name="b")

expr = tf.matmul(X,W) + b

print("--- X ---")
print(X)
print("--- W ----")
print(W)
print("--- b ---")
print(b)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("--- W ---")
print(sess.run(W))
print("--- b ---")
print(sess.run(b))
print("--- expr ---")
print(sess.run(expr, feed_dict={X: x_data}))

sess.close()