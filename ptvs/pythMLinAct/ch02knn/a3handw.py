#!/usr/bin/env python3
# a3handw.py
""" k-近邻算法实战之sklearn手写识别系统
(1)收集数据：提供文本文件。
(2)准备数据：编写函数classify0( ) ,将图像格式转换为分类器使用的制格式。
(3)分析数据：在Python命令提示符中检查数据，确保它符合要求。
(4)训练算法：此步驟不适用于各近邻算法。
(5)测试算法：编写函数使用提供的部分数据集作为测试样本
(6)使用算法：本例没有完成此步驟
"""
from os import listdir
import numpy  as np
import operator


'''将32x32的二进制图像转换为1x1024向量
为了使用前面两个例子的分类器，我们必须将图像格式化处理为一个向量
'''
def img2vector(filename):
    returnVect = np.zeros((1,1024)) #创建1x1024零向量
    fr = open(filename)
    for i in range(32):  # 文件有32行，按行读取
        lineStr = fr.readline()  # 读一行数据
        for j in range(32):  # 每一行的前32个元素依次添加到returnVect中
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]   #获取dataSet的行数
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet  # A-B
    sqDiffMat = diffMat**2   # 平方
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5   # 距离d，开根
    sortedDistIndicies = distances.argsort()  #按照从小到大的次序排序
    classCount={}          # 定义分类，确定前k个距离最小元素所在的主要分类
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

''' 手写数字识别系统的测试代码
k-近邻算法识别手写数字数据集，错误率为1.2%.
改变变量k的值、修改随机选取训练样本、改变训练样本的数目，
都会对k-近邻算法的错误率产生影响。
'''
def handwritingClassTest():
    hwLabels = []   # 测试集的Labels
    trainingFileList = listdir(r'./ch02knn/trainingDigits')   #返回目录下的文件名
    m = len(trainingFileList)  # 文件数量
    trainingMat = np.zeros((m,1024))  # 创建一个m行1024列的训练矩阵，每行数据存储一个图像
    for i in range(m):
        fileNameStr = trainingFileList[i]  # 获得文件的名字
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0]) # 文件命名规则如 2_78.txt，2为分类数字
        hwLabels.append(classNumStr) #将分类数字添加到hwLabels中
        trainingMat[i,:] = img2vector(r'./ch02knn/trainingDigits/%s' % fileNameStr)
    # 对testDigits目录中的文件使用classify0()函数测试
    testFileList = listdir(r'./ch02knn/testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(r'./ch02knn/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        # print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    #print "\nthe total number of errors is: %d" % errorCount
    #print "\nthe total error rate is: %f" % (errorCount/float(mTest))
    print("总共错了%d个数据\n错误率为%f" % (errorCount, errorCount/float(mTest)))


if __name__ == '__main__':
	handwritingClassTest()
