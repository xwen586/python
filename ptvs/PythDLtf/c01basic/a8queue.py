#!/usr/bin/env python3
# coding:utf-8
# a8queue.py
"""Queue：
tf.FIFOQueue 按入列顺序出列的队列
tf.RandomShuffleQueue 随机顺序出列的队列
tf.PaddingFIFOQueue 以固定长度批量出列的队列
tf.PriorityQueue 带优先级出列的队列
"""
import tensorflow as tf

''' tf.FIFOQueue
'''
def demoFIFO():
    q = tf.FIFOQueue(capacity=3, dtypes="int32")
    init = q.enqueue_many(([1, 2, 3], )) #3个元素排入队列
    x = q.dequeue() # 第1个元素出队
    y = x + 10
    q_inc = q.enqueue([y]) #重新加入队列
    with tf.Session() as sess:
        sess.run(init)
        quelen = sess.run(q.size()); print(quelen)
        for i in range(5):
            x1, y1, _ = sess.run([x, y, q_inc]) # 执行2 次操作，队列中的值变为0.3,1.1,1.2
            print(x1, y1)


'''------- 随机队列 tf.RandomShuffleQueue -------
'''
def demoRandom():
    #队列最大长度为10，出队后最小长度为2
    q = tf.RandomShuffleQueue(capacity=10, min_after_dequeue=2, dtypes="float")
    with tf.Session() as sess:
        for i in range(0, 10): #10 次入队
            sess.run(q.enqueue(i+1))
        try:
            # 10次出队，大于最小长度，形成阻断; 设置等待时间5s来解除阻断
            for i in range(0, 10):
                run_options = tf.RunOptions(timeout_in_ms = 5000)
                print(sess.run(q.dequeue(), options=run_options), end="  ") # 随机出8个值
        except tf.errors.DeadlineExceededError:
            print('out of range')


''' 主程序 '''
if __name__=='__main__':
    demoFIFO();    print('-'*20)
    demoRandom();  print('-'*20)
