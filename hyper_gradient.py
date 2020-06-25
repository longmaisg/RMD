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
X_eval = np.random.rand(4, 10)
Y = np.sqrt(np.sum(X, axis=1)).T.reshape(4, 1)
Y_eval = np.sqrt(np.sum(X_eval, axis=1)).T.reshape(4, 1)
print('X:', X)
print('Y:', Y)

# neural network
inner_lr = tf.Variable(1e-3)
outer_lr = tf.constant(5e-3)
x = tf.placeholder(dtype=tf.float32, shape=[4, 10])
y = tf.placeholder(dtype=tf.float32, shape=[4, 1])
w = tf.Variable(tf.random_normal(shape=[10, 1], mean=0, stddev=0.05))
theta = tf.Variable(tf.random_normal(shape=[10, 10], mean=0, stddev=0.05))


# prediction and loss
def predict(x, w, inner_lr, theta, is_backprop=False):
    y_hat = tf.matmul(tf.matmul(x, theta), w)
    loss = tf.reduce_sum(tf.square(y - y_hat))

    if is_backprop:
        var_list = [w]
        grad_w = tf.gradients(loss, var_list)
        assign_list = []
        for g, v in zip(grad_w, var_list):
            assign_list.append(v.assign(v - inner_lr * g[0]))
        train_step = tf.group(*assign_list)
        return loss, train_step
    else:
        return y_hat


pred = predict(x, w, inner_lr, theta, is_backprop=False)
loss, train_step = predict(x, w, inner_lr, theta, is_backprop=True)


def evaluate(x_eval, y_eval, w_ph, inner_lr, theta):
    y_hat = tf.matmul(x_eval, w_ph)
    loss = tf.reduce_sum(tf.square(y_eval - y_hat))
    d_w = tf.gradients(loss, [w])[0]
    return d_w, loss


d_w_eval, outer_loss = evaluate(x, y, w, inner_lr, theta)


def reverse(x, y, w, inner_lr, theta, d_w, d_inner_lr, d_theta):
    y_hat = tf.matmul(tf.matmul(x, theta), w)
    loss = tf.reduce_sum(tf.square(y - y_hat))
    g_t = tf.gradients(loss, [w])[0]
    d_inner_lr = d_inner_lr - tf.reduce_sum(d_w * g_t)
    d_theta = d_theta - inner_lr * d_w * tf.gradients(g_t, [theta])[0]
    d_w = d_w * (1 - inner_lr * tf.gradients(g_t, [w])[0])
    return d_w, d_inner_lr, d_theta


d_w_ph = tf.placeholder(dtype=tf.float32, shape=[10, 1])
d_inner_lr_ph = tf.placeholder(dtype=tf.float32, shape=())
d_theta_ph = tf.placeholder(dtype=tf.float32, shape=(10, 10))
w_ph = tf.placeholder(dtype=tf.float32, shape=[10, 1])
d_w, d_inner_lr, d_theta = reverse(x, y, w_ph, inner_lr, theta, d_w_ph, d_inner_lr_ph, d_theta_ph)


def update_hyper_parameters(inner_lr, d_inner_lr, theta, d_theta, outer_lr):
    assign_list = []
    assign_list.append(inner_lr.assign(inner_lr - outer_lr * d_inner_lr))
    assign_list.append(theta.assign(theta - outer_lr * d_theta))
    train_step = tf.group(*assign_list)
    return train_step


update_hyper = update_hyper_parameters(inner_lr, d_inner_lr_ph, theta, d_theta_ph, outer_lr)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    l_all = []
    lr_all = []
    init_w = sess.run(w)
    for i in range(200):
        # inner-loop:
        x_all = []
        y_all = []
        w_all = []
        # load the initial value of w again
        w.load(init_w, sess)
        num_inner_loop = 20
        for t in range(0, num_inner_loop, 1):
            # save values for Reverse-mode differentiation
            _w = sess.run(w)
            w_all.append(_w)
            x_all.append(X)
            y_all.append(Y)

            l, _ = sess.run([loss, train_step], feed_dict={x: X, y: Y})

        # evaluation:
        _d_w, _outer_loss = sess.run([d_w_eval, outer_loss], feed_dict={x: X_eval, y: Y_eval})
        l_all.append(round(_outer_loss, 4))
        _d_inner_lr = 0
        _d_theta = np.zeros(shape=(10, 10))

        # outer loop
        for t in range(num_inner_loop-1, -1, -1):
            feed_dict = {x: x_all[t], y: y_all[t], w_ph: w_all[t], d_w_ph: _d_w, d_inner_lr_ph: _d_inner_lr, d_theta_ph: _d_theta}
            _d_w, _d_inner_lr, _d_theta = sess.run([d_w, d_inner_lr, d_theta], feed_dict=feed_dict)

        # update hyper-parameters
        sess.run(update_hyper, feed_dict={d_inner_lr_ph: _d_inner_lr, d_theta_ph: _d_theta})

        # get the loss and lr
        _lr = sess.run(inner_lr)
        lr_all.append(_lr)
        print("epoch: %d" % i, "\touter_loss: %.4f" % _outer_loss, "\tlr: %.4f" % _lr)

        # control value of lr:
        if abs(_lr) > 0.01:
            inner_lr.load(0.01, sess)
        if abs(_lr) < 1e-4:
            inner_lr.load(5e-4, sess)

        # stop if loss increases again (overfitting)
        if (len(l_all) > 10) and (l_all[-1] >= l_all[-2]):
            break

    print("pred:", sess.run(pred, feed_dict={x: X, y: Y}))
    print("Y:", Y)

import matplotlib.pyplot as plt
plt.plot(l_all)
plt.show()
plt.plot(lr_all)
plt.show()
