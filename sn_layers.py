#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Modified based on 
# https://github.com/minhnhat93/tf-SNDCGAN

import numpy as np
import tensorflow as tf

from sn import spectral_normed_weight

def conv2d(inputs, 
    out_dim, k_size, strides,
    padding="SAME",
    w_init=None,
    use_bias=True, 
    spectral_normed=True, 
    name="conv2d",
    ):

    with tf.variable_scope(name):

        w = tf.get_variable("w", 
            shape=[k_size, k_size, inputs.get_shape()[-1], out_dim], 
            dtype=tf.float32,
            initializer=w_init
            )
        
        if spectral_normed:
            w = spectral_normed_weight(w)
        
        conv = tf.nn.conv2d(inputs, w, strides=[1, strides, strides, 1], padding=padding.upper())
        
        if use_bias:
            biases = tf.get_variable("b", [out_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases, name="conv_add_b")
        
        return conv


def linear(inputs, 
    out_dim, 
    w_init=None,
    activation=None,
    use_bias=True, bias_start=0.0,
    spectral_normed=False, 
    name="linear", 
    ):

    with tf.variable_scope(name):

        w = tf.get_variable("w", 
            shape=[ inputs.get_shape()[-1], out_dim ], 
            dtype=tf.float32,
            initializer=w_init
            )
        
        if spectral_normed:
            w = spectral_normed_weight(w)

        mul = tf.matmul(inputs, w, name='linear_mul')

        if use_bias:
            bias = tf.get_variable("b", [out_dim], initializer=tf.constant_initializer(bias_start))
            mul = tf.nn.bias_add(mul, bias, name="mul_add_b")

        if not (activation is None):
            mul = activation(mul)

        return mul


