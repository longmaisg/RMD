# from __future__ import print_function, division
import numpy as np
import random
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# set random seed
tf.reset_default_graph()
tf.set_random_seed(123)
random.seed(123)
np.random.seed(123)

# input data
X = np.random.rand(4, 3)
Y = np.sqrt(np.sum(X, axis=1)).T.reshape(4, 1)
print('X:', X)
print('Y:', Y)

# neural network
lr = tf.constant(0.01)
x = tf.placeholder(dtype=tf.float32, shape=[4, 3])
y = tf.placeholder(dtype=tf.float32, shape=[4, 1])
w = tf.Variable(tf.random_normal(shape=[3, 1], mean=0, stddev=0.05))

# prediction and loss
pred = tf.matmul(x, w)
loss = tf.reduce_sum(tf.square(y - pred))

# # method 1
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
# update_w = optimizer.minimize(loss)

# # method 2
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
# var_list = [w]
# grad_w = optimizer.compute_gradients(loss, var_list)[0]
# grads_and_vars = zip(grad_w, var_list)
# update_w = optimizer.apply_gradients(grads_and_vars)

# method 3
var_list = [w]
grad_w = tf.gradients(loss, var_list)[0]
update_w = w - lr * grad_w
w_update = tf.placeholder(dtype=tf.float32, shape=[3, 1])
assign_op_w = w.assign(w_update)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    l_all = []
    for i in range(50):
        l, u_w = sess.run([loss, update_w], feed_dict={x: X, y: Y})
        l_all.append(l)
        print(l)
        # method 3
        sess.run([assign_op_w], feed_dict={w_update: u_w})

    print("pred:", sess.run(pred, feed_dict={x: X, y: Y}))
    print("Y:", Y)

import matplotlib.pyplot as plt
plt.plot(l_all)
plt.show()

