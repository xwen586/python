#!/usr/bin/env python3
# a3modrest.py
""" 保存和加载模型参数
仅做变量的存储和加载；
对 a2modsave.py 中存储的w, u做加载

https://blog.csdn.net/zywvvd/article/details/77941680 存储为不同类型数据
保存的检查点文件:
.meta文件保存了当前图结构
.index文件保存了当前参数名
.data文件保存了当前参数值
"""
import tensorflow as tf
import numpy as np


# 定义变量，矩阵类型，由后续随机数填充
aW = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name="weights")
aB = tf.Variable([[1,2,3]], dtype=tf.float32, name="biases")

cp_dir = r'./c02saver/mod/a3/' # checkpoints文件的保存路径
saver = tf.train.Saver()

# 保存
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 初始化变量
    save_path = saver.save(sess, cp_dir+'a3.ckpt')
    print("Save to path:",save_path)

sess.close()  #关闭上段的Session

# 提取
print("提取", "-"*20)
#rW = tf.Variable(np.arange(6).reshape((2,3)), dtype=tf.float32, name="weights")
#rB = tf.Variable(np.arange(3).reshape((1,3)), dtype=tf.float32, name="biases")
saver2 = tf.train.Saver()
with tf.Session() as sess2:  #提取上边存储的 a3.ckpt
    saver2.restore(sess2, cp_dir+'a3.ckpt')
    print("weights:",sess2.run(aW))
    print("biases:",sess2.run(aB))

#提取a2modsave.py中存储的model.ckpt
saver3 = tf.train.Saver()
with tf.Session() as sess3:
    ckpt = tf.train.get_checkpoint_state(r'./c02saver/mod/a2/')
    if ckpt and ckpt.model_checkpoint_path:
        saver3.restore(sess3, ckpt.model_checkpoint_path)
    print('提取a2：w=[%f], u=[%f]' % (sess2.run('weight:0'), sess2.run('bias:0')) ) #版本2


