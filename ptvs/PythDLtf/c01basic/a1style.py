#!/usr/bin/env python3
# a1scope.py
""" 设计理念
"""
import tensorflow as tf

'''--------- 符号式计算 --------- '''
t = tf.add(8, 9)  #等同 t=8+9
print(t) # 输出 Tensor("Add:0", shape=(), dtype=int32)

'''TensorFlow 中涉及的运算都要放在图中，而图的运行只发生在会话
http://wiki.jikexueyuan.com/project/tensorflow-zh/get_started/basic_usage.html
1)使用图 (graph) 来表示计算任务.
2)在被称之为 会话 (Session) 的上下文 (context) 中执行图.
3)使用 张量(tensor) 表示数据.
4)通过 变量 (Variable) 维护状态.
5)使用 feed 和 fetch 可以为任意的操作(arbitrary operation) 赋值或者从其中获取数据.
https://www.cnblogs.com/denny402/p/5852083.html
'''
#1.常量
a = tf.constant(10)

#2.变量
x = tf.Variable(tf.ones([3,3]))  
y = tf.Variable(tf.zeros([3,3]))
v = tf.Variable(tf.constant(5.0, shape=[1], name='v') #与get_variable等价
v1= tf.get_variable('v', shape=[1], initializer=tf.constant(5.0))#constant_initializer
z = tf.get_variable(name="z", initializer=tf.constant(2))#通过变量的名字来使用变量
# 变量定义完后，还必须显式的执行一下初始化操作
init=tf.global_variables_initializer()

#3.占位符
x = tf.placeholder(tf.float32, [None, 784])

#4.创建图(graph)。
a = tf.constant([1.0, 2.0])  #创建各个节点
b = tf.constant([3.0, 4.0])
c = a * b  # 
sess = tf.Session()# 创建会话 (Session)
print(sess.run(c)) # 计算c, 进行矩阵乘法，输出[3., 8.]
sess.close()

#创建一个变量op, 产生一个 1x2 矩阵. 这个op被作为一个节点加到默认图中.
m1= tf.Variable([[3, 4]])   # 1×2 矩阵
m2= tf.Variable([[1],[2]])  # 2×1 矩阵
d = tf.matmul(m1, m2)  #矩阵乘法运算
# 建议书写形式
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())# 不做初始化会出错
    print(sess.run(d))
print(res)

# ---- Feed ----
# 使用一个tensor值临时替换一个操作的输出结果.
state = tf.Variable(0, name="counter") #初始化为标量0
input = tf.constant(3.0)  # 常量张量
input1 = tf.placeholder(tf.float32)  #占位符 填充数据
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))#Feed操作

#字符串变量操作
word=tf.constant('hello,world!')
print("word:",word)
with tf.Session() as sess:
    print("word:",sess.run(word))

#矩阵乘法
a=tf.Variable(tf.ones([3,2]))
b=tf.Variable(tf.ones([2,3]))
product=tf.matmul(5*a,4*b)
#init=tf.initialize_all_variables()
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(product))


# 交互式使用，为了便于使用诸如 IPython 之类的 Python 交互环境
sess = tf.InteractiveSession()
x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])
x.initializer.run()# 使用初始化器的 run() 方法初始化 'x' 
# 增加一个减法 sub op, 从 'x' 减去 'a'. 运行减法 op, 输出结果 
sub = tf.subtract(x, a)
print(sub.eval())


# ---- Tensor ----
# 使用 张量(Tensor)表示数据。计算图中, 操作间传递的数据都是tensor
# 例：使用变量实现一个简单的计数器
state = tf.Variable(0, name="counter") #创建一个变量, 初始化为标量0.
one = tf.constant(1)     # 创建一个op, 加1用
new_value = tf.add(state, one)       #累加
update = tf.assign(state, new_value) #更新tensor.给变量赋初值
with tf.Session() as sess:  # 启动图, 运行 op
    sess.run(tf.global_variables_initializer()) #初始化
    print(sess.run(state) )
    for _ in range(3):  # 运行 op, 更新 'state', 并打印 'state'
        sess.run(update)
        print(sess.run(state))


# ---- Fetch ----
# 取回操作的输出内容。
# 例：取回多个tensor
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
interm = tf.add(input2, input3)  # Fetch1
mul = tf.multiply(input1, interm)# Fetch2
with tf.Session() as sess:
    result = sess.run([mul, interm]) #fetch操作
    print(result)





