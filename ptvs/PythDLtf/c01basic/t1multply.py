#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# t1multply.py
""" 简单乘法
https://github.com/nlintz/TensorFlow-Tutorials/blob/master/00_multiply.py
"""
import tensorflow as tf

a = tf.placeholder("float") # Create a symbolic variable 'a'
b = tf.placeholder("float") # Create a symbolic variable 'b'
y = tf.multiply(a, b) # multiply the symbolic variables

with tf.Session() as sess: # create a session to evaluate the symbolic expressions
    y1 = sess.run(y, feed_dict={a: 1, b: 2})
    y2 = sess.run(y, feed_dict={a: 3, b: 3})
    print("%f should equal 2.0" % y1) # eval expressions with parameters for a and b
    print("%f should equal 9.0" % y2)

