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
lr = tf.Variable(0.01)
x = tf.placeholder(dtype=tf.float32, shape=[4, 3])
y = tf.placeholder(dtype=tf.float32, shape=[4, 1])
w = tf.Variable(tf.random_normal(shape=[3, 1], mean=0, stddev=0.05))

# prediction and loss
pred = tf.matmul(x, w)
loss = tf.square(y - pred)
sum_loss = tf.reduce_sum(loss)

# method 3 - update weights
var_list = [w]
grad_w = tf.gradients(loss, var_list)[0]
update_w = w - lr * grad_w
w_update = tf.placeholder(dtype=tf.float32, shape=[3, 1])
assign_op_w = w.assign(w_update)

# update lr
pred_t1 = tf.matmul(x, update_w)
loss_t1 = tf.square(y - pred_t1)
sum_loss_t1 = tf.reduce_sum(loss_t1)

alpha = tf.constant(0.0001)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
var_list_lr = [lr]
grad_lr = tf.gradients(loss_t1, var_list_lr)
grads_and_vars_lr = zip(grad_lr, var_list_lr)
update_lr = optimizer.apply_gradients(grads_and_vars_lr)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    l_all = []
    lr_all = []
    for i in range(50):
        _w_t = sess.run(w)
        _lr, l, _grad_w_t, _w_update, _ = sess.run([lr, sum_loss, grad_w, update_w, update_lr], feed_dict={x: X, y: Y})
        l_all.append(l)
        lr_all.append(_lr)
        print(l, '\t', _lr)

        sess.run([assign_op_w], feed_dict={w_update: _w_update})

    print("pred:", sess.run(pred, feed_dict={x: X, y: Y}))
    print("Y:", Y)

import matplotlib.pyplot as plt
plt.plot(l_all)
plt.show()

plt.plot(lr_all)
plt.show()
