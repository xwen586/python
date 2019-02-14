#!/usr/bin/env python3
# coding:utf-8
# q9queueRunner.py
""" 队列管理器

"""
import tensorflow as tf
import numpy as np


''' 队列管理器 tf.QueueRunner
QueueRunner管理读写队列线程的。
'''
def demoQueueRunner():
    q = tf.FIFOQueue(100, "float")
    counter = tf.Variable(0.0) # 计数器
    inc_op = tf.assign_add(counter, tf.constant(1.0)) #操作：给计数器加1
    enq_op = q.enqueue(counter) #操作：计数器值加入队列
    # 队列管理器,  操作inc_op,enq_op向队列q中添加元素，两个操作不同步
    qrun = tf.train.QueueRunner(q, enqueue_ops=[inc_op, enq_op] * 1) # *1 一个线程
    # 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        enqueue_threads = qrun.create_threads(sess, start=True) # 启动入队线程
        #主线程 与入队线程是异步的，会一直等待数据送入
        for i in range(10):
            print(sess.run(q.dequeue()), end="--> ")
        #其他线程会继续运行，程序不会结束。


''' 协调器 tf.Coordinator
Coordinator是个用来保存线程组运行状态的协调器对象，
它和TensorFlow的Queue没有必然关系，是可以单独和Python线程使用的。
'''
import threading, time
def demoCoordinator():
    # 主线程
    coord = tf.train.Coordinator()
    # 使用Python API创建10个子线程
    threads = [threading.Thread(target=loop, args=(coord, i)) for i in range(10)]
    # 启动所有线程，并等待线程结束
    for t in threads: t.start()
    coord.join(threads)
    #主线程会等待所有子线程都停止后结束，从而使整个程序结束。
    print("main thread stop!")

# 子线程函数
def loop(coord, id):
    t = 0
    #线程通过should_stop方法感知并停止当前线程。
    while not coord.should_stop():
        print("sub thread[%d]: t=%d" % (id, t) )
        time.sleep(0.3) #0.3s
        t += 1
        # 只有9号线程调用request_stop方法
        if (t >= 2 and id == 9):
            coord.request_stop()



''' 线程和协调器
tf.Coordinator和tf.QueueRunner从设计上这两个类必须被一起使用。
'''
def demoQRwithCoord():
    q = tf.FIFOQueue(100, "float")
    counter = tf.Variable(0.0) # 计数器
    inc_op = tf.assign_add(counter, tf.constant(1.0)) #操作：给计数器加1
    enq_op = q.enqueue(counter) #操作：计数器值加入队列
    qrun = tf.train.QueueRunner(q, enqueue_ops=[inc_op, enq_op] * 1) # *1 一个线程
    # 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Coordinator：协调器，协调线程间的关系可以视为一种信号量，用来做同步
        coordi = tf.train.Coordinator()
        enqueue_threads = qrun.create_threads(sess=sess, coord=coordi, start=True) # 启动入队线程
        coordi.request_stop()# 通知其他线程关闭
        for i in range(10):
            try:
                print (sess.run(q.dequeue()), end="  ")
            except tf.errors.OutOfRangeError:
                print("Do not dequeue!"); break
        # join操作等待其他线程结束，其他所有线程关闭之后，这一函数才能返回
        coordi.join(enqueue_threads)


'''------- QueueRunner和Coordinator一起使用 -------

'''
def demo1():
    # 1000个4维输入向量，每个数取值为1-10之间的随机数
    data = np.random.randn(1000, 4)
    # 1000个随机的目标值，值为0或1
    #target = np.random.randint(0, 2, size=1000)
    target = np.arange(1, 1001) #
    print("init data of 100: ", data[ 99:100,], target[ 99:100])
    print("init data of 200: ", data[199:200,], target[199:200])

    # 创建Queue，队列中每一项包含一个输入数据和相应的目标值
    queue = tf.FIFOQueue(capacity=5, dtypes=[tf.float32, tf.int32], shapes=[[4], []])
    # 批量入列数据（这是一个Operation）
    enqueue_op = queue.enqueue_many([data, target]) #超出队列容量，线程停滞，等待消费。
    # 出列数据（这是一个Tensor定义）
    data_sample, label_sample = queue.dequeue()
    # 创建包含4个线程的QueueRunner, 4个入队线程
    qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)

    with tf.Session() as sess:
        # 创建Coordinator
        coord = tf.train.Coordinator()
        # 启动QueueRunner管理的线程
        enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
        # 主线程，消费100个数据
        for step in range(200):
            if coord.should_stop():
                break
            data_batch, label_batch = sess.run([data_sample, label_sample])
            if label_batch == 100: coord.request_stop()  #可设置在第100次停止线程。
        print(data_batch, label_batch)
        # 主线程计算完成，停止所有采集数据的进程
        coord.request_stop()
        coord.join(enqueue_threads)


''' 主程序 '''
if __name__=='__main__':
    #demoQueueRunner()
    #demoCoordinator()
    #demoQRwithCoord()
    demo1()
