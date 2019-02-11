#!/usr/bin/env python
# t2lr.py
""" 线性回归 优化器Optimizer演示
 Y = wX + b
https://github.com/nlintz/TensorFlow-Tutorials/blob/master/01_linear_regression.py
"""
import tensorflow as tf
import numpy as np

# 训练数据集，创建一个近似于线性，带随机噪声的y值。
trX = np.linspace(-1, 1, 101)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33 # create a y value which is approximately linear but with some random noise

X = tf.placeholder("float") # create symbolic variables
Y = tf.placeholder("float")


def model(X, w):
    return tf.multiply(X, w) # lr is just X*w so this model line is pretty simple


w = tf.Variable(0.0, name="weights") # create a shared variable (like theano.shared) for the weight matrix
y_model = model(X, w)

cost = tf.square(Y - y_model) #用方差作为成本函数  use square error for cost function

#构造一个优化器，使成本最小，并与数据拟合。
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # construct an optimizer to minimize cost and fit line to my data

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize variables (in this case just variable W)
    tf.global_variables_initializer().run()

    for i in range(100):
        for (x, y) in zip(trX, trY):
            sess.run(train_op, feed_dict={X: x, Y: y})
    w_value=sess.run(w)
    print(w_value)  # It should be something around 2

# 图形展示
import matplotlib.pyplot as plt
plt.plot(trX, trY, "+")
rY= trX.dot(w_value)
plt.plot(trX, rY)
plt.show()
