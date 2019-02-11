#!/usr/bin/env python3
# exmp02.py
""" 一个简单的计算图
https://zhuanlan.zhihu.com/p/37259681
1) 每次执行一个函数
2）执行 tensorboard --host=127.0.0.1 --port=8081 --logdir="logs"
3）访问  http://localhost:8081
"""
import tensorflow as tf

''' 常量加法
未定义图名称
'''
def addgraph():
    a = 2
    b = 3
    c = tf.add(a, b, name='Add')
    print(c)

    with tf.Session().as_default() as sess:
        writer = tf.summary.FileWriter(r"./tensorboard/logs/", sess.graph)
        print(sess.run(c))
        writer.close()


''' 常量
'''
def constgph():
    a = tf.constant(2, name='a')
    b = tf.constant(3, name='b')
    x = tf.constant(4)
    y = tf.constant(5)
    c = a + b + x * y
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(r"./tensorboard/logs/", sess.graph)
        print(sess.run(c))
        writer.close()


''' 变量
'''
def vargph():
    a = tf.get_variable(name="a", initializer=tf.constant(2))
    b = tf.get_variable(name="b", initializer=tf.constant(3))
    m1= tf.Variable([[3, 4]])   # 1×2 矩阵
    m2= tf.Variable([[1],[2]])  # 2×1 矩阵
    c = tf.add(a, b, name="Add1")
    d = tf.matmul(m1, m2)
    e = tf.add(c, d, name="Add2")
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(r"./tensorboard/logs/", sess.graph)
        sess.run(tf.global_variables_initializer())
        print(sess.run(e))
        writer.close()


'''占位符, Feed, Fetch
'''
def placegph():
    x = tf.placeholder(tf.float32, name='x')  #占位符 填充数据
    y = tf.placeholder(tf.float32)
    v = tf.Variable(3.14, tf.float32, name="v")
    c = tf.constant(3.0)  # 常量
    v1= tf.add(v, c, name="Add")
    update = tf.assign(v, v1)
    output = tf.multiply(x+v, y)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(r"./tensorboard/logs/", sess.graph)
        sess.run(tf.global_variables_initializer()) #初始化
        print(sess.run([output, update], feed_dict={x:[7.], y:[2.]}))#Feed操作
        writer.close()


''' 主程序 '''
if __name__=='__main__':
    #addgraph()
    #constgph()
    #vargph()
    placegph()