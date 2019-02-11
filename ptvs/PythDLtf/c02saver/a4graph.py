#!/usr/bin/env python3
# a4graph.py
""" 图存储（Meta graph）
Graph 被定义为“一些 Operation 和 Tensor 的集合”。
从python Graph中序列化出来的图就叫做 GraphDef。
graph声明有三种：https://blog.csdn.net/zj360202/article/details/78539464 
1.tensor:通过张量本身直接出graph
2.声明一个默认的graph，然后定义张量内容，在后面可以调用或保存
3.声明多个graph，在后面通过变量名来分别调用

对graph的操作:
"""
import tensorflow as tf
import numpy as np

'''----- 通过张量本身获取graph -----
'''
def defaultCreate():
    c = tf.constant(np.pi)   # tensor
    g = tf.get_default_graph()
    assert c.graph is g #看看主程序中新建的一个变量是不是在默认图里
    with tf.Session().as_default() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(c))
        print(g)
        print(c.graph)


'''----- 自定义graph，并作为默认图 -----
'''
def defineGraph():
    with tf.Graph().as_default() as g1:  # 定义了g1并作为默认图
        c2 = tf.constant(np.pi*2)
        v2= tf.Variable(tf.random_normal(shape=(1,)), dtype = tf.float32, name='var2')
    with tf.Session(graph=g1) as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(c2), sess.run(v2))
        print("get_default_graph:", tf.get_default_graph())
        print("c2.graph:", c2.graph)
        print("g1:", g1)


'''----- 声明多个graph -----
'''
def defineMoreGraph():
    with tf.Graph().as_default() as g1:  # 注意 as_default(),此后的张量都为该graph的
        c1 = tf.constant(5.0)
    with tf.Graph().as_default() as g2:
        c2 = tf.constant(20.0)
    with tf.Session(graph=g1) as sess1:
        print("at g1:", sess1.run(c1), g1)
    with tf.Session(graph=g2) as sess2:
        print("at g2:", sess2.run(c2), g2)


'''----- 穿插调用 -----
两个不同graph下的张量操作 
'''
def usebyCross():
    with tf.Graph().as_default() as g1:  # 注意 as_default(),此后的张量都为该graph的
        c1 = tf.constant(5.0, name='c1')
    with tf.Graph().as_default() as g2:
        c2 = tf.constant(20.0, name='c2')
    with tf.Session(graph=g2) as sess1:
        # 通过名称和下标来得到相应的值
        c1_list = tf.import_graph_def(g1.as_graph_def(), return_elements = ["c1:0"], name = '')
        print(sess1.run(c1_list[0]+c2))


'''----- 对graph的操作 -----
 1.保存
 2.从pb文件中加载，二进制和文本方式
 3.穿插调用
'''
#
def saveGraph():
    g = tf.get_default_graph()
    print(g)
    with tf.Graph().as_default() as g1:
        c1 = tf.constant(5.0, name='c1')
    with tf.Graph().as_default() as g2:
        c2 = tf.constant(20.0, name='c2')
    with tf.Session(graph=g1) as sess1:
        print(sess1.run(c1), g1)
    with tf.Session(graph=g2) as sess2:
        print(sess2.run(c2), g2)
    # 保存g1: 图的定义，包含pb的path, pb文件名，是否是文本默认False
    tf.train.write_graph(g1.as_graph_def(), r'./c02saver/graph/', 'g1.pb', False) #二进制写
    # 保存g2: 文本方式
    tf.train.write_graph(g2.as_graph_def(), r'./c02saver/graph/', 'g2.pb', True) #文本写


# 2.从pb文件中加载
def loadGraph():
    with tf.gfile.GFile(r"./c02saver/graph/g1.pb", 'rb') as f: #二进制读取
        graph_def = tf.GraphDef()  
        graph_def.ParseFromString(f.read())  
        output = tf.import_graph_def(graph_def, name='')
        print(output)
    print(graph_def)
    with tf.Session() as sess:
        c1 = sess.graph.get_tensor_by_name("c1:0")  #获取tensor
        print(sess.run(c1))

# 用write_graph写pb时，令as_text=True 的读取
# 见 https://www.jb51.net/article/138784.htm 最后
from google.protobuf import text_format  #引入文本格式转换
def loadGraph2():
    #with open(r"./c02saver/graph/g2.pb", 'r') as f:
    with tf.gfile.GFile(r"./c02saver/graph/g2.pb", 'r') as f: #与open()相同
        readf = f.read()
    graph_def = tf.GraphDef()
    text_format.Merge(readf, graph_def) #不用graph_def.ParseFromString(f.read())
    with tf.Session() as sess:
        c2 = tf.import_graph_def(graph_def, return_elements=['c2:0']) 
        print(sess.run(c2))


# 读取不同的graph下存储的数据
def demo01():
    with tf.Graph().as_default() as g1:
        with tf.gfile.GFile(r"./c02saver/graph/g1.pb", 'rb') as f: #二进制读取
            graph_def1 = tf.GraphDef()
            graph_def1.ParseFromString(f.read())
        c1 = tf.import_graph_def(graph_def1, return_elements=['c1:0'])# 通过名称和下标来得到相应的值

    with tf.Graph().as_default() as g2:
        with tf.gfile.GFile(r"./c02saver/graph/g2.pb", 'r') as f: #文本读取
            readf = f.read()
        graph_def2 = tf.GraphDef()  
        # 不使用graph_def.ParseFromString(f.read())
        text_format.Merge(readf, graph_def2) #转换写入
        tf.import_graph_def(graph_def2, name='')

    with tf.Session(graph=g1) as sess1:        
        c1v= sess1.run(c1)
        print("c1=", c1v, c1)
    with tf.Session(graph=g2) as sess2:        
        c2 = sess2.graph.get_tensor_by_name("c2:0")  #获取tensor，另一方法
        print("c1+c2=",sess2.run(c2 + c1v), c2)



''' 主程序 '''
if __name__ == '__main__':
    #defaultCreate(); print('-'*20)  #通过张量本身获取graph
    #defineGraph();   print('-'*20)  #自定义graph，并作为默认图
    #defineMoreGraph(); print('-'*20)#声明多个graph
    #usebyCross()      ; print('-'*20)#穿插调用
    #saveGraph()      ; print('-'*20)#保存graph
    #loadGraph()      ; print('-'*20)#从pb文件中加载
    #loadGraph2()      ; print('-'*20)#文本方式从pb文件中加载
    demo01()
