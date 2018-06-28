import tensorflow as tf
import numpy as np

#data 6 
#input 2 -> 2,10 -> 10,3 -> 3
x_data = np.array([[0,0], [1,0], [1,1], [0,0], [0,0], [0,1]]) 
y_data = np.array([[1,0,0], [0,1,0], [0,0,1], [1,0,0], [1,0,0], [0,0,1]])

W1 = tf.Variable(tf.random_uniform([2,10], -1, 1))
W2 = tf.Variable(tf.random_uniform([10,3], -1, 1))

b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([3]))

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

L1 = tf.add(tf.matmul(X, W1), b1)
R1 = tf.nn.relu(L1)
L2 = tf.add(tf.matmul(R1, W2), b2)
R2 = tf.nn.relu(L2)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=R2))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range (100):
    sess.run(train_op, feed_dict={X:x_data, Y:y_data})
    
    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(loss, feed_dict={X:x_data, Y:y_data}))
        print(sess.run(W1))

prediction = tf.argmax(R2, 1)
answer = tf.argmax(Y, 1)
print("Prediction: ", sess.run(prediction, feed_dict={X:x_data}))
print("Answer    : ", sess.run(answer, feed_dict={Y:y_data}))

is_correct = tf.equal(prediction, answer)
accracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("Accuracy: %.2f" % sess.run(accracy * 100, feed_dict={X:x_data, Y:y_data}))
