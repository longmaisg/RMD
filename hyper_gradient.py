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
X = np.random.rand(4, 10)
Y = np.sqrt(np.sum(X, axis=1)).T.reshape(4, 1)
print('X:', X)
print('Y:', Y)

# neural network
alpha = tf.Variable(0.01)
theta = tf.Variable(0.1)
meta_lr = tf.constant(0.0001)
x = tf.placeholder(dtype=tf.float32, shape=[4, 10])
y = tf.placeholder(dtype=tf.float32, shape=[4, 1])
w = tf.Variable(tf.random_normal(shape=[10, 1], mean=0, stddev=0.05))


# prediction and loss
def predict(x, w, alpha, theta, is_backprop=False):
    y_hat = tf.matmul(x, w)
    loss = tf.reduce_sum(tf.square(y - y_hat + theta))

    if is_backprop:
        var_list = [w]
        grad_w = tf.gradients(loss, var_list)
        assign_list = []
        for g, v in zip(grad_w, var_list):
            assign_list.append(v.assign(v - alpha * g[0]))
        train_step = tf.group(*assign_list)
        return loss, train_step
    else:
        return y_hat


pred = predict(x, w, alpha, theta, is_backprop=False)
loss, train_step = predict(x, w, alpha, theta, is_backprop=True)


def evaluate(x_eval, y_eval, w_ph, alpha, theta):
    y_hat = tf.matmul(x_eval, w_ph)
    loss = tf.reduce_sum(tf.square(y_eval - y_hat + theta))
    d_w = tf.gradients(loss, [w])[0]
    return d_w


d_w_eval = evaluate(x, y, w, alpha, theta)


def reverse(x, y, w, alpha, theta, d_w, d_alpha, d_theta):
    y_hat = tf.matmul(x, w)
    loss = tf.reduce_sum(tf.square(y - y_hat + theta))
    g_t = tf.gradients(loss, [w])[0]
    d_alpha = d_alpha - tf.squeeze(tf.matmul(tf.transpose(d_w), g_t))
    # d_theta = d_theta - alpha * tf.gradients(g_t, [theta])[0]
    d_theta = d_theta - alpha * tf.reduce_sum(d_w) * tf.gradients(g_t, [theta])[0]
    d_w = d_w * (1 - alpha * tf.gradients(g_t, [w])[0])
    return d_w, d_alpha, d_theta


d_w_ph = tf.placeholder(dtype=tf.float32, shape=[10, 1])
d_alpha_ph = tf.placeholder(dtype=tf.float32, shape=())
d_theta_ph = tf.placeholder(dtype=tf.float32, shape=())
w_ph = tf.placeholder(dtype=tf.float32, shape=[10, 1])
d_w, d_alpha, d_theta = reverse(x, y, w_ph, alpha, theta, d_w_ph, d_alpha_ph, d_theta_ph)


def update_hyper_parameters(alpha, d_alpha, theta, d_theta, meta_lr):
    assign_list = []
    assign_list.append(alpha.assign(alpha - meta_lr * d_alpha))
    assign_list.append(theta.assign(theta - meta_lr * d_theta))
    train_step = tf.group(*assign_list)
    return train_step


update_hyper = update_hyper_parameters(alpha, d_alpha_ph, theta, d_theta_ph, meta_lr)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    l_all = []
    lr_all = []
    for i in range(10):
        print(sess.run([alpha, theta]))
        # inner-loop:
        x_all = []
        y_all = []
        w_all = []
        for t in range(0, 5, 1):
            # save values for Reverse-mode differentiation
            _w = sess.run(w)
            w_all.append(_w)
            x_all.append(X)
            y_all.append(Y)

            l, _ = sess.run([loss, train_step], feed_dict={x: X, y: Y})
            l_all.append(l)
            print(l)

        # evaluation:
        _d_w = sess.run(d_w_eval, feed_dict={x: X, y: Y})
        _d_alpha = 0
        _d_theta = 0

        # outer loop
        for t in range(5-1, -1, -1):
            feed_dict = {x: x_all[t], y: y_all[t], w_ph: w_all[t], d_w_ph: _d_w, d_alpha_ph: _d_alpha, d_theta_ph: _d_theta}
            _d_w, _d_alpha, _d_theta = sess.run([d_w, d_alpha, d_theta], feed_dict=feed_dict)

        # update hyper-parameters
        sess.run(update_hyper, feed_dict={d_alpha_ph: _d_alpha, d_theta_ph: _d_theta})

    print("pred:", sess.run(pred, feed_dict={x: X, y: Y}))
    print("Y:", Y)

import matplotlib.pyplot as plt
plt.plot(l_all)
plt.show()
