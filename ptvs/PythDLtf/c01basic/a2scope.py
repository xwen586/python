#!/usr/bin/env python3
# a2scope.py
"""
作用域
分为 name_scope（命名空间），variable_scope（变量作用域）
"""
import tensorflow as tf


'''
tf.name_scope（命名空间）
输出：
name1/weights:0
name2/weights:0
'''
with tf.name_scope('name1') as scope:
    weights1 = tf.Variable([1.0, 2.0], name='weights')
    bias1 = tf.Variable([0.3], name='bias')

with tf.name_scope('name2') as scope:
    weights2 = tf.Variable([4.0, 2.0], name='weights')
    bias2 = tf.Variable([0.33], name='bias')

print (weights1.name)  # name1/weights:0
print (weights2.name)  # name2/weights:0


'''
tf.variable_scope（变量作用域）
'''
with tf.variable_scope('v_scope') as scope1:
   Weights1 = tf.get_variable('Weights', shape=[2, 3])
   bias1 = tf.get_variable('bias', shape=[3])

# note: 在下面的 scope 中的get_variable()变量必须已经定义过了，
#才能设置 reuse=True，否则会报错
with tf.variable_scope('v_scope', reuse=True) as scope2:
   Weights2 = tf.get_variable('Weights')
   Weights3 = tf.get_variable('Weights', [2, 3]) #shape如果不同会报错
print (Weights2.name)  # v_scope/Weights:0
print (Weights3.name)  # v_scope/Weights:0


