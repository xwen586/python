#!/usr/bin/env python3
# exmp01.py
''' TensorBoard
样例一
1）在有TensorFlow环境下执行本程序：python exmp01.py
2）执行成功时，生成logs目录
3）执行 tensorboard --host=127.0.0.1 --port=8081 --logdir="logs"
4）访问  http://localhost:8081
'''
import tensorflow as tf

with tf.name_scope('graph') as scope:
     matrix1 = tf.constant([[3., 3.]],name ='matrix1')  #1 row by 2 column
     matrix2 = tf.constant([[2.],[2.]],name ='matrix2') # 2 row by 1 column
     product = tf.matmul(matrix1, matrix2,name='product')
  
sess = tf.Session()

writer = tf.summary.FileWriter(r"./tensorboard/logs/", sess.graph)

init = tf.global_variables_initializer()

sess.run(init)
