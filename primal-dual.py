#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 22:09:44 2019

@author: hliang
"""

def update_lagrangian(u_, l, level, iter):
    assert len(u_) == len(l)

    with tf.name_scope("lagrange_update"):
        with tf.variable_scope("step{}".format(iter),reuse=True):
            sig = tf.get_variable("sig")
        sum_u = tf.reduce_sum(u_[level], axis=4, keep_dims=False)
        l[level] += sig * (sum_u - 1.0)


def update_dual(u_, m, steps, level, iter):
    assert len(u_) == len(m)

    with tf.name_scope("dual_update"):
        with tf.variable_scope("weights_step{}_{}".format(steps, level), reuse=True):
            w1 = tf.get_variable("w1")
            w2 = tf.get_variable("w2")

        _, nrows, ncols, nslices, nclasses = \
            u_[level].get_shape().as_list()
        batch_size = tf.shape(u_[level])[0]

        if level + 1 < len(u_):
            grad_u1 = conv3d(u_[level], w1)
            grad_u2 = conv3d(u_[level + 1], w2)
            grad_u = grad_u1 + resize_volumes(grad_u2, 2, 2, 2)
        else:
            grad_u = conv3d(u_[level], w1)
            
        with tf.variable_scope("step{}".format(iter), reuse=True):
            sig = tf.get_variable("sig")

        m[level] += sig * grad_u

        m_rshp = tf.reshape(m[level], [batch_size, nrows, ncols,
                                       nslices, nclasses, 3])

        m_norm = tf.norm(m_rshp, ord="euclidean", axis=5, keep_dims=True)
        m_norm = tf.maximum(m_norm, 1.0)
        m_normalized = tf.divide(m_rshp, m_norm)

        m[level] = tf.reshape(m_normalized, [batch_size, nrows, ncols,
                                             nslices, nclasses * 3])


def update_primal(u, m, l, d, steps, level, iter):
    assert len(u) == len(m)
    assert len(u) == len(l)
    assert len(u) == len(d)

    with tf.name_scope("primal_update"):
        with tf.variable_scope("weights_step{}_{}".format(steps, level), reuse=True):
            w1 = tf.get_variable("w1")
            w2 = tf.get_variable("w2")

        # u_shape = tf.shape(u[level])
        # batch_size = u_shape[0]
        # nrows = u_shape[1]
        # ncols = u_shape[2]
        # nslices = u_shape[3]
        # nclasses = u_shape[4]

        _, nrows, ncols, nslices, nclasses = \
            u[level].get_shape().as_list()
        batch_size = tf.shape(u[level])[0]

        if level + 1 < len(u):
            div_m1 = conv3d_adj(m[level], w1, nclasses)
            div_m2 = conv3d_adj(m[level + 1], w2, nclasses)
            div_m = div_m1 + resize_volumes(div_m2, 2, 2, 2)
        else:
            div_m = conv3d_adj(m[level], w1, nclasses)

        l_rshp = tf.reshape(l[level], [batch_size, nrows, ncols, nslices, 1])
        with tf.variable_scope("step{}".format(iter),reuse=True):
            tau = tf.get_variable("tau")
        u[level] -= tau * (d[level] + l_rshp + div_m)

        u[level] = tf.minimum(1.0, tf.maximum(u[level], 0.0))


def primal_dual(u, u_, m, l, d, steps, iter):
    u_0 = list(u)

    for level in list(range(len(u)))[::-1]:
        with tf.name_scope("primal_dual_iter{}_level{}".format(iter, level)):
            update_dual(u_, m, steps, level, iter)

            update_lagrangian(u_, l, level, iter)

            update_primal(u, m, l, d, steps, level, iter)

            u_[level] = 2 * u[level] - u_0[level]

    return u, u_, m, l
