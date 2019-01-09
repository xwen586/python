#!/usr/bin/env python3
# a1knn.py
""" k-近邻算法，k-Nearest Neighbors
"""

from numpy import *
import operator

''' 
 使用欧氏距离公式，计算两个向量点A和B之间的距离
 有4个输人参数：用于分类的输入向量是inX，输入的训练样本集为dataSet,
 标签向量为labels，最后的参数k表示用于选择最近邻居的数目
'''
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]   #获取dataSet的行数
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  # A-B
    sqDiffMat = diffMat**2   # 平方
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5   # 距离d，开根
    sortedDistIndicies = distances.argsort()  #按照从小到大的次序排序
    classCount={}          # 定义分类，确定前k个距离最小元素所在的主要分类
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # Python 3.x 里面，iteritems()已经废除，用 items()替代
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 创建数据集和标签
def createDataSet():
    group = array([ [1.0,1.1], [1.0,1.0], [0,0], [0,0.1] ])
    labels = ['A','A','B','B']
    return group, labels


if __name__ == '__main__':
    group, labels = createDataSet()
    ix = [0, 0]
    iy = [0.8, 0.9]
    resultx = classify0( ix, group, labels, 3) #输出结果应该是B,  ix靠近B
    print('x点' + str(ix) +' 临近：' + resultx)
    resulty = classify0( iy, group, labels, 3) #输出结果应该是B,  ix靠近B
    print('y点' + str(iy) +' 临近：' + resulty)
