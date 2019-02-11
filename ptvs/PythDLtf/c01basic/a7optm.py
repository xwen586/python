#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# a7optm.py
"""4.7.5 优化方法 
目前加速训练的优化方法基本都是基于梯度下降.
大多数机器学习任务就是最小化损失，在损失定义的情况下，后面的工作就交给优化器.
TensorFlow 提供了很多优化器（optimizer）:
class tf.train.GradientDescentOptimizer  #批梯度下降法（BGD 和SGD）
class tf.train.AdadeltaOptimizer  # Adadelta法 自适应优化器
class tf.train.AdagradOptimizer   # Adagrad法
class tf.train.AdagradDAOptimizer 
class tf.train.MomentumOptimizer  # Momentum 法
class tf.train.AdamOptimizer
class tf.train.FtrlOptimizer
class tf.train.RMSPropOptimizer

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


'''--------- 批梯度下降法 GradientDescentOptimizer ---------
BGD 的全称是batch gradient descent,
是利用现有参数对训练集中的每一个输入生成一个估计输出yi，然后跟实际输出yi比较，
统计所有误差，求平均以后得到平均误差，以此作为更新参数的依据。
原文：https://blog.csdn.net/xierhacker/article/details/53174558
'''
# Prepare train data
train_X = np.linspace(-1, 1, 100)
# 直线: y = 2.0x + 2.0
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 2

# Define the model
X = tf.placeholder("float")
Y = tf.placeholder("float")
w = tf.Variable(0.0, name="weight") #权重
b = tf.Variable(0.0, name="bias") #偏差
loss = tf.square(Y - X*w - b)  #损失函数 Y-tf.multiply(X,w)-b
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# Create session to run
with tf.Session() as sess:
    #sess.run(tf.initialize_all_variables())
    tf.global_variables_initializer().run()
    epoch = 1
    for i in range(20): #迭代越多越精确
        for (x, y) in zip(train_X, train_Y): #zip对各list组合
            _, w_value, b_value = sess.run([train_op, w, b],feed_dict={X: x,Y: y})
        print("Epoch: {}, w: {}, b: {}".format(epoch, w_value, b_value))
        if epoch==1 : w1=w_value; b1=b_value #取第一次优化结果
        if epoch==4 : w2=w_value; b2=b_value #取第4次优化结果
        epoch += 1

#draw
plt.plot(train_X, train_Y, "+")
plt.plot(train_X, train_X.dot(w1)+b1, color='g')  #第一次拟合线
plt.plot(train_X, train_X.dot(2) +2, color='blue')#原始拟合线
fitY = train_X.dot(w_value)+b_value  # 最终拟合线
plt.plot(train_X, fitY, color='r')
plt.show()


'''--------- 正弦曲线拟合 AdamOptimizer ---------
Adam 的名称来源于自适应矩估计（adaptive moment estimation）。
Adam法根据损失函数针对每个参数的梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。
https://blog.csdn.net/piaoxuezhong/article/details/78907069
'''
#
datasize= 100
train_X = np.linspace(0, 4*np.pi, datasize)
train_Y = np.sin(train_X) + 1

# Define the model
X1 = tf.placeholder(tf.float32,shape=(datasize,))
X2 = tf.placeholder(tf.float32,shape=(datasize,))
X3 = tf.placeholder(tf.float32,shape=(datasize,))
X4 = tf.placeholder(tf.float32,shape=(datasize,))
Y  = tf.placeholder(tf.float32,shape=(datasize,))
w1 = tf.Variable(0.0, name="weight1")
w2 = tf.Variable(0.0, name="weight2")
w3 = tf.Variable(0.0, name="weight3")
w4 = tf.Variable(0.0, name="weight4")

y1 = w1*X1 + w2*X2 + w3*X3 + w4*X4
loss = tf.reduce_mean(tf.square(Y - y1))

optimizer =tf.train.AdamOptimizer() #use adam method to optimize
train_op = optimizer.minimize(loss)

# Create session to run
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(5000):
        _, ww1, ww2, ww3, ww4, loss_= sess.run([train_op, w1, w2,w3,w4,loss],\
            feed_dict={X1:train_X, X2:train_X**3, X3:train_X**5, X4:train_X**7, Y:train_Y})

plt.plot(train_X, train_Y, "+", label='data')
fitY = ww1*train_X+(ww2)*(train_X**3)+ww3*(train_X**5)+ww4*(train_X**7)  # 最终拟合线
plt.plot(train_X, fitY, label='curve')
#plt.savefig('1.png',dpi=200)
#plt.axis([0,np.pi,-2,2])
plt.legend(loc=1)
plt.show()
