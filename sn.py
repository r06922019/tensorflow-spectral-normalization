#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Modified based on 
# https://github.com/pfnet-research/sngan_projection/issues/15#issuecomment-406939990
# https://github.com/minhnhat93/tf-SNDCGAN

import tensorflow as tf
import numpy as np 

def spectral_normed_weight(w, 
    u=None, 
    num_iters=1, # For Power iteration method, usually num_iters = 1 will be enough
    update_collection=None, 
    with_sigma=False # Estimated Spectral Norm
    ):

    w_shape = w.shape.as_list()
    w_new_shape = [ np.prod(w_shape[:-1]), w_shape[-1] ]
    w_reshaped = tf.reshape(w, w_new_shape, name='w_reshaped')
    
    if u is None:
        u = tf.get_variable("u_vec", [w_new_shape[0], 1], initializer=tf.truncated_normal_initializer(), trainable=False)
    
    # power iteration
    u_ = u
    for _ in range(num_iters):
        # ( w_new_shape[1], w_new_shape[0] ) * ( w_new_shape[0], 1 ) -> ( w_new_shape[1], 1 )
        v_ = _l2normalize(tf.matmul(tf.transpose(w_reshaped), u_)) 
        # ( w_new_shape[0], w_new_shape[1] ) * ( w_new_shape[1], 1 ) -> ( w_new_shape[0], 1 )
        u_ = _l2normalize(tf.matmul(w_reshaped, v_))

    u_final = tf.identity(u_, name='u_final') # ( w_new_shape[0], 1 )
    v_final = tf.identity(v_, name='v_final') # ( w_new_shape[1], 1 )

    u_final = tf.stop_gradient(u_final)
    v_final = tf.stop_gradient(v_final)

    sigma = tf.matmul(tf.matmul(tf.transpose(u_final), w_reshaped), v_final, name="est_sigma")

    update_u_op = tf.assign(u, u_final)

    with tf.control_dependencies([update_u_op]):
        sigma = tf.identity(sigma)
        w_bar = tf.identity(w / sigma, 'w_bar')

    if with_sigma:
        return w_bar, sigma
    else:
        return w_bar

def _l2normalize(v, eps=1e-12):
    with tf.name_scope('l2normalize'):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)
