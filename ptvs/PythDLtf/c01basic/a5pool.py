#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# a5pool.py
""" 池化函数
池化函数一般跟在卷积函数的下一层
它们也被定义在tensorflow-1.1.0/tensorflow/python/ops下的nn.py 和gen_nn_ops.py 文件中。
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/ops
"""
import tensorflow as tf
import os
import numpy as np

#log 日志级别设置:
# 1-默认的显示等级，显示所有信息;
# 2-只显示 warning 和 Error 
# 3-只显示 Error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#
input_data  = tf.Variable(np.random.rand(10, 9, 9, 3), dtype=np.float32)
filter_data = tf.Variable(np.random.rand( 2, 2, 3, 2), dtype=np.float32)
y = tf.nn.conv2d(input_data, filter_data, strides=[1, 1, 1, 1], padding='SAME')
print('0. tf.nn.conv2d : ', y)

# 计算池化区域中元素的平均值
output = tf.nn.avg_pool(value=y, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
print('1. tf.nn.avg_pool : ', output)

# 计算池化区域中元素的最大值
output = tf.nn.max_pool(value=y, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
print('2. tf.nn.max_pool : ', output)

# 计算池化区域中元素的最大值,与最大值所在位置
# 1.1.0似乎只支持GPU,本代码首测运行于 python3.6.2 + Tensorflow(CPU) 1.2.0 + win10
output, argmax = tf.nn.max_pool_with_argmax(input=y, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
print('2.5 . tf.nn.max_pool : ', output, argmax)

# 与conv2d_transpose 二维反卷积类似
# 在解卷积网络(deconvolutional network) 中有时被称为'反卷积',但实际上是conv3d的转置,而不是实际的反卷积
input_data = tf.Variable(np.random.rand(1, 2, 5, 5, 1), dtype=np.float32)
filters = tf.Variable(np.random.rand(2, 3, 3, 1, 3), dtype=np.float32)
y = tf.nn.conv3d(input_data, filters, strides=[1, 2, 2, 1, 1], padding='SAME')
print('3. tf.nn.conv3d : ', y)

# 计算三维下池化区域中元素的平均值
output = tf.nn.avg_pool3d(input=y, ksize=[1, 1, 2, 2, 1], strides=[1, 2, 2, 1, 1], padding='SAME')
print('4. tf.nn.avg_pool3d : ', output)

# 计算三维下池化区域中元素的最大值
output = tf.nn.max_pool3d(input=y, ksize=[1, 1, 2, 2, 1], strides=[1, 2, 2, 1, 1], padding='SAME')
print('5. tf.nn.max_pool3d : ', output)

# 执行一个N维的池化操作
# def pool(input, window_shape,pooling_type,padding,dilation_rate=None,strides=None,name=None,data_format=None):

