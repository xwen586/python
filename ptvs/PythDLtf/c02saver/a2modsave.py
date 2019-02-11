#!/usr/bin/env python3
# a2modsave.py
""" 模型存储与加载
训练好一个模型后，把模型存储起来，在预测数据时加载使用。
建立一个tf.train.Saver()来保存变量，并且指定保存的位置，一般模型的扩展名为.ckpt
本例 对训练中的 w、u 存储，加载。
"""
import tensorflow as tf
import numpy as np

'''
说明：
版本1(V1)中没有定义train_model(), load_model()函数, 是连续使用变量,有相同上下文。
版本2(V2)定义函数，进行隔离来验证模型加载。
'''
# 训练模型
def train_model(cdir):
    # 原型定义（ Y = aX + b ）并训练
    x = tf.placeholder(tf.float32, shape=[None, 1])
    a = tf.constant(2.8)
    b = tf.constant(6.9)
    y = a * x + b

    # 学习后的拟合参数
    #运行时初始化一随机数，由random_normal从正太分布的数值中取出指定个数的值。
    w = tf.Variable(tf.random_normal([1], -1, 1), name="weight") # 
    u = tf.Variable(tf.zeros([1]), name="bias") # 初始值为0
    y_predict = w * x + u

    loss = tf.reduce_mean(tf.square(y - y_predict)) #损失函数
    optimizer = tf.train.GradientDescentOptimizer(0.5) #优化器
    train = optimizer.minimize(loss)

    train_steps = 100      #训练的次数
    checkpoint_steps = 50  #训练多少次保存一下checkpoints
    checkpoint_dir = cdir #r'./c02saver/mod/a2/' # checkpoints文件的保存路径

    x_data = np.random.rand(10).astype(np.float32).reshape(10, 1) # 随机生成10个数据
    saver = tf.train.Saver() #声明tf.train.Saver类用于保存模型

    with tf.Session() as sess:
        #sess.run(tf.initialize_all_variables()) # 在V1.9中没有用了。
        sess.run(tf.global_variables_initializer()) # 初始化全部变量
        for i in range(train_steps):
            sess.run(train, feed_dict={x: x_data})
            if (i + 1) % checkpoint_steps == 0:
                saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=i+1)
                print('训练%d：w= [%f], u= [%f]' % (i+1, sess.run(w), sess.run(u)) )

#print(sess)
#sess.close()  #关闭上段的Session

''' 此处关注：
1）版本1 运行时，w,u等变量是全局的，saver2可以找到对应的存储点，加载保存的数据。
2）版本2使用load_model()进行上下文隔离。
  与train_model()连续运行时，saver2声明前可以不定义变量w,u; 使用'weight:0'寻找加载点。
  load_model()独立运行时，saver2声明前需定义变量w,u（否则No variables to save）;
'''
# 加载模型，与train_model()连续运行用
def load_model1(cdir):
    checkpoint_dir = cdir
    # 此处不定义变量，否则报错：Key bias_1 not found in checkpoint
    #w = tf.Variable(tf.zeros([1]), name="weight") # 
    #u = tf.Variable(tf.zeros([1]), name="bias") # 初始值为0
    saver2 = tf.train.Saver()
    # 测试
    with tf.Session() as sess2: 
        # restore时，不需要进行init= tf.initialize_all_variables()操作。
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver2.restore(sess2, ckpt.model_checkpoint_path) #恢复加载模型
        else:
            pass  # 跳过
        #print('测试：w=[%f], u=[%f]' % (sess2.run(w), sess2.run(u)) ) # 版本1 重复使用w,u的定义
        print('测试：w=[%f], u=[%f]' % (sess2.run('weight:0'), sess2.run('bias:0')) ) #版本2


# 加载模型，独立运行
def load_model2(cdir):
    checkpoint_dir = cdir
    # 若不定义变量，报：ValueError: No variables to save
    # 相当于：模型重新定义一遍
    w = tf.Variable(tf.truncated_normal(shape=(1,)), dtype = tf.float32, name='weight') 
    b = tf.Variable(tf.truncated_normal(shape=(1,)), dtype = tf.float32, name='bias') 
    saver2 = tf.train.Saver()
    with tf.Session() as sess2: 
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            tf.train.latest_checkpoint(checkpoint_dir)
            saver2.restore(sess2, ckpt.model_checkpoint_path) #恢复加载模型
        else:
            pass  # 跳过
        print('加载（带变量定义）：w=[%f], u=[%f]' % (sess2.run(w), sess2.run(b)) ) # 版本1 重复使用w,u的定义


# 不需重新定义模型，使用 import_meta_graph()
def load_model3(cdir):
    checkpoint_dir = cdir
    with tf.Session() as sess3: 
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
            saver.restore(sess3, ckpt.model_checkpoint_path) #恢复加载模型
        else:
            pass  # 跳过
        #print('测试：w=[%f], u=[%f]' % (sess3.run(w), sess3.run(u)) )
        # 通过张量的名称来获取张量
        print('测试：w=[%f], u=[%f]' % (sess3.run('weight:0'), sess3.run('bias:0')) )
        print(sess3.run(tf.get_default_graph().get_tensor_by_name("weight:0")))


''' 主程序'''
if __name__ == '__main__':
    checkpoint_dir = r'./c02saver/mod/a2/' # checkpoints文件的保存路径
    # 模型训练与加载
    #train_model(checkpoint_dir) #训练
    print("-"*20)    
    #load_model1(checkpoint_dir) # 模型加载

    # 模型独立加载
    #load_model2(checkpoint_dir)
    # 模型独立加载,不需重新定义模型
    load_model3(checkpoint_dir)
