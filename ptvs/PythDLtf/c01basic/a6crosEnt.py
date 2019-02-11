#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# a6crosEnt.py
""" 分类函数（交叉熵的计算）
主要有sigmoid_cross_entropy_with_logits、softmax、log_softmax、softmax_cross_entropy_with_logits 等
https://www.cnblogs.com/guqiangjs/p/8202899.html
"""
import tensorflow as tf


''' ------------ tf.nn.sigmoid_cross_entropy_with_logits
该函数计算的是 logits 与 lables 的每一个对应维度上对应元素的损失值。数值越小，表示损失值越小。
'''
_logits = [[0.5, 0.7, 0.3], [0.8, 0.2, 0.9]]
_one_labels = tf.ones_like(_logits)
_zero_labels = tf.zeros_like(_logits)
with tf.Session() as sess:
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=_logits, labels=_one_labels)
    # [[0.47407699  0.40318602  0.5543552]
    #  [0.37110069  0.59813887  0.34115386]]
    print(sess.run(loss))

    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=_logits, labels=_zero_labels)
    # [[0.97407699  1.10318601  0.85435522]
    #  [1.17110074  0.79813886  1.24115384]]
    print(sess.run(loss))


''' ------------ tf.nn.softmax_cross_entropy_with_logits
该函数与 sigmoid_cross_entropy_with_logits 的区别在于，
sigmoid_cross_entropy_with_logits 中的labels 中每一维可以包含多个1，而
softmax_cross_entropy_with_logits ，只能包含一个 1。
'''
_logits = [[0.3, 0.2, 0.2], [0.5, 0.7, 0.3], [0.1, 0.2, 0.3]]
_labels = [0, 1, 2]

with tf.Session() as sess:
    # Softmax本身的算法很简单，就是把所有值用e的n次方计算出来，求和后算每个值占的比率，保证总和为1，一般我们可以认为Softmax出来的就是confidence也就是概率
    # [[0.35591307  0.32204348  0.32204348]
    #  [0.32893291  0.40175956  0.26930749]
    #  [0.30060959  0.33222499  0.36716539]]
    print(sess.run(tf.nn.softmax(_logits)))
    # 对 _logits 进行降维处理，返回每一维的合计
    # [1.  1.  0.99999994]
    print(sess.run(tf.reduce_sum(tf.nn.softmax(_logits), 1)))

    # 传入的 lables 需要先进行 独热编码 处理。
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=_logits, labels=tf.one_hot(_labels,depth=len(_labels)))
    # [ 1.03306878  0.91190147  1.00194287]
    print(sess.run(loss))



''' ------------  tf.one_hot
独热编码
'''
print('------ tf.one_hot ------ ')
with tf.Session() as sess:
    _v = tuple(range(0, 5))
    # [[ 1.  0.  0.  0.  0.]
    #  [ 0.  1.  0.  0.  0.]
    #  [ 0.  0.  1.  0.  0.]
    #  [ 0.  0.  0.  1.  0.]
    #  [ 0.  0.  0.  0.  1.]]
    print(sess.run(tf.one_hot(_v, len(_v))))



''' ------------ sparse_softmax_cross_entropy_with_logits

sparse_softmax_cross_entropy_with_logits 是 softmax_cross_entropy_with_logits 的易用版本，
除了输入参数不同，作用和算法实现都是一样的。
'''
print('------ sparse_softmax_cross_entropy_with_logits ------ ')
_logits = [[0.3, 0.2, 0.2], [0.5, 0.7, 0.3], [0.1, 0.2, 0.3]]
_labels = [0, 1, 2]
with tf.Session() as sess:
    # 
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=_logits, labels=_labels)
    # 结果与 softmax_cross_entropy_with_logits 相同，区别就是 labels 传入参数时不需要做 one_hot encoding。
    # [ 1.03306878  0.91190147  1.00194287]
    print(sess.run(loss))


''' ------------ softmax，log_softmax

Softmax本身的算法很简单，就是把所有值用e的n次方计算出来，求和后算每个值占的比率，保证总和为1.
一般我们可以认为Softmax出来的就是confidence也就是概率
该功能执行相当于：
   softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), dim)
logsoftmax = logits - log(reduce_sum(exp(logits), dim))
'''
print('------ softmax，log_softmax ------ ')
with tf.Session() as sess:
    _v = tf.Variable(tf.random_normal([1, 5], seed=1.0))
	#当训练自己的神经网络的时候，无一例外的就是都会加上一句
    sess.run(tf.global_variables_initializer())

    # [[-0.81131822  1.48459876  0.06532937 -2.4427042   0.0992484]]
    print(sess.run(_v))
    # [[0.06243069  0.6201579   0.15001042  0.01221508  0.15518591]]
    print('softmax: ' + sess.run(tf.nn.softmax(_v)))
    # 1.0
    print('reduce_sum: ', sess.run(tf.reduce_sum(tf.nn.softmax(_v))))
	# [[-2.773698   -0.47778115 -1.8970505  -4.405084   -1.8631315 ]]
    print('log_softmax: ', sess.run(tf.nn.log_softmax(_v)))

