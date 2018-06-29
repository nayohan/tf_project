import tensorflow as tf
import numpy as np

x_data = np.array([[0,0], [1,0], [1,1], [0,0], [0,0], [0,1]]) 
y_data = np.array([[1,0,0], [0,1,0], [0,0,1], [1,0,0], [1,0,0], [0,0,1]])

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

W = tf.Variable(tf.random_uniform([2, 3], -1, 1))
b = tf.Variable(tf.zeros([3]))

affine = tf.add(tf.matmul(X, W), b)
relu = tf.nn.relu(affine)
model = tf.nn.softmax(relu)

loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(1000):
    train_rslt = sess.run(train_op, feed_dict={X: x_data, Y: y_data})
    loss_rslt  = sess.run(loss, feed_dict={X: x_data, Y: y_data})
    
    if (step + 1) % 100 == 0:
        print(step + 1, loss_rslt)
        print(sess.run(W))
        print(sess.run(b))

prediction = tf.argmax(model, axis=1)
answer = tf.argmax(Y, axis=1)
print("Predict : ", sess.run(prediction, feed_dict = {X: x_data}))
print("Answer  : ",  sess.run(answer, feed_dict = {Y: y_data}))

is_correct = tf.equal(prediction, answer)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("Accuracy : %.2f" % sess.run(accuracy * 100, feed_dict = {X: x_data, Y: y_data}))

sess.close()    
