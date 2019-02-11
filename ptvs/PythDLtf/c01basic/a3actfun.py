#!/usr/bin/env python3
# a3actfun.py
""" 激活函数
Sigmoid、tanh 、relu 、relu6、 elu 、softplus 
"""
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


'''-------------- sigmoid()，tanh() --------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
def tanh(x):
    return 2 * sigmoid(2*x) - 1
'''
print("---- sigmoid()，tanh()")
a = tf.constant([[1.0, 0.2], [1.5, 2.0], [2.0, 4.0]])
sess = tf.Session()
print( sess.run(tf.sigmoid(a)) )
print( sess.run(tf.tanh(a)) )
sess.close()

x = np.linspace(-10, 10)
with tf.Session() as sess:
    y = sess.run(tf.sigmoid(x))
    t = sess.run(tf.tanh(x))


#画图
matplotlib.rcParams['axes.unicode_minus']=False
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
plt.xlim(-11, 11)
plt.ylim(-1.1, 1.1)

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.set_xticks([-10, -5, 0, 5, 10])
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
ax.set_yticks([-1, -0.5, 0.5, 1])

plt.plot(x*2, y, label="Sigmoid", color = "blue")
plt.plot(x*2, t, label="Tanh",  color = "red")
plt.legend()
plt.show()


'''-------------- ReLU()，softplus()-relu变体 -------------- 
def relu(x): y = np.where(x<0,0,x)
'''
print("---- ReLU()，softplus()")
# relu(), softplus()
a = tf.constant([-1.0, 2.0])
with tf.Session() as sess:
    b = tf.nn.relu(a)
    print(sess.run(b))
	#
    c = tf.nn.softplus(a)
    print(sess.run(c))


x = np.linspace(-10, 10)
with tf.Session() as sess:
    y = sess.run(tf.nn.relu(x))
    s = sess.run(tf.nn.softplus(x))


#画图
matplotlib.rcParams['axes.unicode_minus']=False
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
plt.xlim(-11,11)
plt.ylim(-5, 5)

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.set_xticks([-10, -5, 0, 5, 10])
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
ax.set_yticks([-5, -2, 2, 5])

plt.plot(x, y, label="ReLU", color="blue")
plt.plot(x, s, label="softplus", color = "red")
plt.legend()
plt.show()


'''-------------- dropout 函数 -------------- 
def dropout(x): 
'''
print("---- dropout()")
a = tf.constant([[-1.0, 2.0, 3.0, 4.0]])
with tf.Session() as sess:
    d = tf.nn.dropout(a, 0.5, noise_shape = [1,4])
    d1= sess.run(d)
    print(d1)
    d = tf.nn.dropout(a, 0.5, noise_shape = [1,1])
    d2= sess.run(d)
    print(d2)


#画图
#matplotlib.rcParams['axes.unicode_minus']=False
#fig = plt.figure(figsize=(6,4))
#ax = fig.add_subplot(111)
#plt.xlim(-11,11)
#plt.ylim(-5, 5)

#ax.spines['top'].set_color('none')
#ax.spines['right'].set_color('none')
#ax.xaxis.set_ticks_position('bottom')
#ax.spines['bottom'].set_position(('data',0))
#ax.set_xticks([-1.0, 2.0, 3.0, 4.0, 5.0])
#ax.yaxis.set_ticks_position('left')
#ax.spines['left'].set_position(('data',0))
#ax.set_yticks([-5, -2, 2, 5])

#plt.plot(a, d1, label="noise[1,4]", color="blue")
#plt.plot(a, d2, label="noise[1,1]", color = "red")
#plt.legend()
#plt.show()


'''-------------- eLU, ReLU6, softsign函数 -------------- 
def dropout(x): 
bias_add()是 tf.add 的一个特例
bias_add(value, bias, data_format=None, name=None)) #将偏差bias加到value上，但最后一维保持一致。
'''
print("---- eLU(), ReLU6(), softsign()")
# tf.add 与 tf.nn.bias_add 区别
a = tf.constant([[1,1], [2,2], [3,3]], dtype=tf.float32)
b = tf.constant([1,-1], dtype=tf.float32)
c = tf.constant([1],dtype=tf.float32)


with tf.Session() as sess:
    print('bias_add:')
    print(sess.run(tf.nn.bias_add(a, b)))
	#执行下面语句错误
    #print(sess.run(tf.nn.bias_add(a, c)))
    print('add:')
    print(sess.run(tf.add(a, b)))
    print(sess.run(tf.add(a, c)))


x = np.linspace(-10, 10)
sess = tf.Session()

e = sess.run(tf.nn.elu(x))
c = sess.run(tf.nn.crelu(x))
r = sess.run(tf.nn.relu6(x))
s = sess.run(tf.nn.softsign(x))


#画图
matplotlib.rcParams['axes.unicode_minus']=False
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
plt.xlim(-11,11)
plt.ylim(-5, 5)

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.set_xticks([-10, -5, 0, 5, 10])
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
ax.set_yticks([-5, -2, 2, 5])

plt.plot(x, e, label="eLU", color="blue")
#plt.plot(x, c, label="CReLU", color="green")
plt.plot(x, r, label="ReLU6", color="brown")
plt.plot(x, s, label="softsign", color = "red")
plt.legend()
plt.show()
