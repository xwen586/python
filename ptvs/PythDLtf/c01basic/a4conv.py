#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# a4conv.py
""" 卷积函数
http://www.cnblogs.com/wanshuai/p/9209498.html
神经网络之所以能解决非线性问题（如语音、图像识别），本质上就是激活函数加入了非线性因素，
弥补了线性模型的表达力，把“激活的神经元的特征”通过函数保留并映射到下一层。
"""
import tensorflow as tf
import numpy as np
import os

#log 日志级别设置:
# 1-默认的显示等级，显示所有信息;
# 2-只显示 warning 和 Error 
# 3-只显示 Error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
卷积函数输入的参数:(input, filter, strides, padding, use_cudnn_on_gpu=None,data_format=None, name=None)
1. input  为输入,一个张量Tensor ,数据类型必须为float32 或者 float64 
2. filter 为卷积核,输入类型必须与input一样 
3. padding为一个字符串取值 SAME为补零使输入输出的图像大小相同,取值VALLD则允许输入输出的图像大小不一致 
4. name,  可选,字符串,用于可视化中,为该操作起一个名字 
5. strides 是另外一个极其重要的参数,其为一个长度为4 的一维整数类型数组,每一位对应input中每一位对应的移动步长

'''

'''-----------1. tf.nn.convolution 计算N维卷积的和-----------
tf.nn.convolution(input, filter, padding, strides=None, dilation_rate=None, name=None,
data_format =None)
'''
# -------- 1. tf.nn.convolution
# 计算N维卷积的和
input_data  = tf.Variable(np.random.rand(10, 9, 9, 3), dtype=np.float32)
filter_data = tf.Variable(np.random.rand( 2, 2, 3, 2), dtype=np.float32)

y1 = tf.nn.convolution(input_data, filter_data, strides=[1, 1], padding='SAME')

print('1. tf.nn.convolution : ', y1)
#  tf.nn.convolution :  Tensor("convolution:0", shape=(10, 9, 9, 2), dtype=float32)


'''-----------2. tf.nn.conv2d()-----------
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None,
name=None)
# 在padding='SAME'时输入输出的图像大小是一致的
# 对一个思维的输入数据 input 和四维的卷积核filter 进行操作,然后对输入的数据进行二维的卷积操作,得到卷积之后的结果
# def conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None,data_format=None, name=None)
# 输入：
# input：一个 Tensor。数据类型必须是 float32 或者 float64
# filter：一个 Tensor。数据类型必须是 input 相同
# strides：一个长度是 4 的一维整数类型数组，每一维度对应的是 input 中每一维的对应移动步数，
# 比如，strides[1]对应 input[1]的移动步数
# padding：一个字符串，取值为 SAME 或者 VALID
# padding='SAME'：仅适用于全尺寸操作，即输入数据维度和输出数据维度相同
# padding='VALID：适用于部分窗口，即输入数据维度和输出数据维度不同
# use_cudnn_on_gpu：一个可选布尔值，默认情况下是 True
# name：（可选）为这个操作取一个名字
# 输出：一个 Tensor，数据类型是 input 相同
'''
input_data = tf.Variable( np.random.rand(10,9,9,3), dtype = np.float32 )
filter_data= tf.Variable( np.random.rand(2, 2,3,2), dtype = np.float32)
y = tf.nn.conv2d(input_data, filter_data, strides = [1, 1, 1, 1], padding = 'SAME')
print('2. tf.nn.conv2d : ', y2)
#2. tf.nn.conv2d :  Tensor("Conv2D:0", shape=(10, 9, 9, 2), dtype=float32)

#with tf.Session() as sess:
#    #y1 = sess.run(y)
#    print(y1)  #Tensor("Conv2D_3:0", shape=(10, 9, 9, 2), dtype=float32)


'''-----------3. tf.nn.depthwise_conv2d()-----------
tf.nn.depthwise_conv2d (input, filter, strides, padding, 
rate=None, name=None,data_format=None)
# input 张量的数据维度 [batch ,in_height,in_wight,in_channels]
# 卷积核的维度是 [filter_height,filter_heught,in_channel,channel_multiplierl]
# 在通道 in_channels 上面的卷积深度是1，depthwise_conv2d 函数将不同的卷积核独立地应用在
# in_channels 的每个通道上（从通道 1到通道 channel_multiplier），然后把所以的结果进行汇总。
# 最后输出通道的总数是 in_channels * channel_multiplier。
'''
input_data = tf.Variable(np.random.rand(10, 9, 9, 3), dtype=np.float32)
filter_data = tf.Variable(np.random.rand(2, 2, 3, 2), dtype=np.float32)
y = tf.nn.depthwise_conv2d(input_data, filter_data, strides=[1, 1, 1, 1], padding='SAME')
print('3. tf.nn.depthwise_conv2d : ', y) #tf.shape(y)的结果是[10 9 9 15]。


'''-----------4. tf.nn.separable_conv2d()-----------
tf.nn.separable_conv2d (input, depthwise_filter, pointwise_filter, strides, padding, 
rate=None, name=None, data_format=None)
# 是利用几个分离的卷积核去做卷积,在该函数中,将应用一个二维的卷积核,
# 在每个通道上,以深度channel_multiplier进行卷积.
# 特殊参数：
# depthwise_filter：一个张量。数据维度是四维[filter_height, filter_width, in_channels,
# channel_multiplier]。其中，in_channels 的卷积深度是 1
# pointwise_filter：一个张量。数据维度是四维[1, 1, channel_multiplier * in_channels,
# out_channels]。其中，pointwise_filter 是在 depthwise_filter 卷积之后的混合卷积
'''
input_data = tf.Variable(np.random.rand(10, 9, 9, 3), dtype=np.float32)
depthwise_filter = tf.Variable(np.random.rand(2, 2, 3, 5), dtype=np.float32)
poinwise_filter = tf.Variable(np.random.rand(1, 1, 15, 20), dtype=np.float32)
# out_channels >= channel_multiplier * in_channels
y = tf.nn.separable_conv2d(input_data, depthwise_filter=depthwise_filter, \
              pointwise_filter=poinwise_filter, strides=[1, 1, 1, 1], padding='SAME')
print('4. tf.nn.separable_conv2d : ', y)  #tf.shape(y)的结果是[10 9 9 20]。


'''-----------5. tf.nn.atrous_conv2d()-----------
# 计算Atrous卷积,又称孔卷积或者扩张卷积

'''
input_data = tf.Variable(np.random.rand(1, 5, 5, 1), dtype=np.float32)
filters = tf.Variable(np.random.rand(3, 3, 1, 1), dtype=np.float32)
y = tf.nn.atrous_conv2d(input_data, filters, 2, padding='SAME')
print('5. tf.nn.atrous_conv2d : ', y)


'''-----------6. tf.nn.conv2d_transpose()-----------
# 在解卷积网络(deconvolutional network) 中有时被称为'反卷积',
但实际上是conv2d的转置,而不是实际的反卷积
'''
x = tf.random_normal(shape=[1, 3, 3, 1])
kernal = tf.random_normal(shape=[2, 2, 3, 1])
y = tf.nn.conv2d_transpose(x, kernal, output_shape=[1, 5, 5, 3], strides=[1, 2, 2, 1], padding='SAME')
print('6. tf.nn.conv2d_transpose: ', y)


'''-----------7. tf.nn.conv1d()-----------
# 与二维卷积类似,用来计算给定三维输入和过滤器的情况下的一维卷积.
# 不同的是,它的输入维度为 3,[batch,in_width,in_channels].
# 卷积核的维度也是三维,[filter_height,in_channel,channel_multiplierl]
# stride 是一个正整数,代表一定每一步的步长
'''
input_data = tf.Variable(np.random.rand(1, 5, 1), dtype=np.float32)
filters = tf.Variable(np.random.rand(3, 1, 3), dtype=np.float32)
y = tf.nn.conv1d(input_data, filters, stride=2, padding='SAME')
print('7. tf.nn.conv1d : ', y)


'''-----------tf.nn.conv3d()-----------
# 与二维卷积类似,用来计算给定五维输入和过滤器的情况下的三维卷积.
# 不同的是,它的输入维度为 5,[batch,in_depth,in_height,in_width,in_channels].
# 卷积核的维度也是三维,[filter_depth,filter_height,in_channel,channel_multiplierl]
# stride 相较二维卷积多了一维,变为[strides_batch,strides_depth,strides_height,strides_width,strides_channel],必须保证strides[0] = strides[4] =1
'''
input_data = tf.Variable(np.random.rand(1, 2, 5, 5, 1), dtype=np.float32)
filters = tf.Variable(np.random.rand(2, 3, 3, 1, 3), dtype=np.float32)
y = tf.nn.conv3d(input_data, filters, strides=[1, 2, 2, 1, 1], padding='SAME')
print('8. tf.nn.conv3d : ', y)



'''-----------9. tf.nn.conv3d_transpose()-----------
# 与conv2d_transpose 二维反卷积类似
# 在解卷积网络(deconvolutional network) 中有时被称为'反卷积',
但实际上是conv3d的转置,而不是实际的反卷积

'''
x = tf.random_normal(shape=[2, 1, 3, 3, 1])
kernal = tf.random_normal(shape=[2, 2, 2, 3, 1])
y = tf.nn.conv3d_transpose(x, kernal, output_shape=[2, 1, 5, 5, 3], strides=[1, 2, 2, 2, 1], padding='SAME')
print('9. tf.nn.conv3d_transpose : ', y)
