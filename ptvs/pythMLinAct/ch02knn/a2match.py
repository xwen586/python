#!/usr/bin/env python3
# a2match.py
"""使用k-近邻算法改进约会网站的配对效果
https://blog.csdn.net/c406495762/article/details/75172850

交往过的人分类：
1）不喜欢的人
2）魅力一般的人
3）极具魅力的人
海伦收集的样本数据主要包含以下3种特征：(datingTestSet.txt文件中）
列1）每年获得的飞行常客里程数  
列2）玩视频游戏所消耗时间百分比
列3）每周消费的冰淇淋公升数
"""
from numpy import *
import operator

'''
 将文本记录到转换NumPy的解析程序
'''
def file2matrix(filename):
    fr = open(filename)  #打开文件
    arrayOLines = fr.readlines()  #读取文件所有内容
    numberOfLines = len(arrayOLines)      #得到文件的行数
    #定义返回的矩阵，存储解析完成的数据:numberOfLines行,3列
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []        #返回的分类标签向量prepare labels return   
    #fr = open(filename)
    index = 0
    for line in arrayOLines:  # fr.readlines():
        line = line.strip()  # 删除空白符
        listFromLine = line.split('\t')  #按'\t'分隔符进行切分
        returnMat[index,:] = listFromLine[0:3]  # 将数据前三列存放到矩阵中,即特征矩阵
        # datingTestSet2.txt用如下语句
        #classLabelVector.append(int(listFromLine[-1]))
        # datingTestSet.txt 用如下语句
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        
        index += 1
    return returnMat, classLabelVector


''' --- 归一化特征值 ---
不同特征值数字差值对计算结果有较大影响，因此采用归一化方法将权重统一。
数值归一化是将特征值转化为0到1区间内的值
newValue = (oldValue-min)/(max-min)
'''
def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 每列的最小值
    maxVals = dataSet.max(0)  # 每列的最大值
    ranges = maxVals - minVals  # 差
    normDataSet = zeros(shape(dataSet))  # shape(dataSet)返回dataSet的矩阵行列数
    m = dataSet.shape[0] #返回dataSet的矩阵行数
    # tile()函数将变量内容(每行)复制成输人矩阵同样大小的矩阵
    normDataSet = dataSet - tile(minVals, (m,1)) # 原始值减去最小值
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals


''' 分类器函数
与 a1knn.py 中完全一致
'''
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  #获取dataSet的行数
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5  # 距离d
    sortedDistIndicies = distances.argsort()   #按照从小到大的次序排序
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


''' 分类器针对约会网站的测试代码
机器学习算法一个很重要的工作就是评估算法的正确率。
提供已有数据的90%作为训练样本来训练分类器，而使用其余的10%数据去测试分类器，
使用错误率来检测分类器的性能，
错误率就是分类器给出错误结果的次数除以测试数据的总数，完美分类器的错误率为0
'''
def datingClassTest():
    hoRatio = 0.20      #hold out 10%
    # 读取数据并将其转换为归一化特征值
    datingDataMat, datingLabels = file2matrix(r'./ch02knn/datingTestSet.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]   # 样本数据的行数
    numTestVecs = int(m*hoRatio)  # 样本数量
    print("the total sample number is: %d" % (numTestVecs))
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
        # print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]) )
        if (classifierResult != datingLabels[i]):
           errorCount += 1.0
           print("No%d. the classifier result: %d, the label is: %d" % (i, classifierResult, datingLabels[i]) )
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)) )


''' ----------- 功能演示 ------------ '''
#导入制图工具
import matplotlib
import matplotlib.pyplot as plt
# 样例1：读取样本文件，解析后散点图显示
def demo01():
    #datingDataMat, datingLabels = file2matrix('kNN/datingTestSet2.txt')
    datingDataMat, datingLabels = file2matrix(r'./ch02knn/datingTestSet.txt')
    fig = plt.figure()
    ax =  fig.add_subplot(111)
    # 矩阵的第二、第三列数据
    # 没有样本类别标签的约会数据散点图。难以辨识图中的点究竟属于哪个样本分类
    ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
    plt.show()

# 样例2：增加标签属性，用色彩、尺寸区分散点图显示
def demo02():
    datingDataMat, datingLabels = file2matrix(r'./ch02knn/datingTestSet.txt')
    fig = plt.figure()
    ax =  fig.add_subplot(111)
    # 利用变量datingLabels存储的类标签属性；矩阵的第二、第三列数据
    ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))  #scatter函数是用来画散点图的
    plt.show()

# 样例3： 归一化特征值，分散点图显示
def demo03():
    datingDataMat, datingLabels = file2matrix(r'./ch02knn/datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    print( 'ranges=%d, minVals=%d ' % (ranges[0], minVals[0]) )
    fig = plt.figure()
    ax =  fig.add_subplot(111)
    ax.scatter(normMat[:,1], normMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))  #scatter函数是用来画散点图的
    plt.show()


"""
函数说明:通过输入一个人的三维特征,进行分类输出
"""
def classifyPerson():
    #输出结果
    resultList = ['讨厌', '有些喜欢', '非常喜欢']
    #三维特征用户输入
    #precentTats = float(input("玩视频游戏所耗时间百分比(10%):"))
    precentTats = 0.1
    a = input("玩视频游戏所耗时间百分比(10)%:")
    if( a!=''): precentTats=float(a)/100
    #ffMiles = float(input("每年获得的飞行常客里程数(5000):"))
    ffMiles = 5000
    a = input("每年获得的飞行常客里程数(5000):")
    if( a!=''): ffMiles=float(a)
    # iceCream = float(input("每周消费的冰激淋公升数(1.2):"))
    iceCream = 1.2
    a = input("每周消费的冰激淋公升数(1.2):")
    if( a!=''): iceCream=float(a)
    #打开并处理数据
    datingDataMat, datingLabels = file2matrix(r'./ch02knn/datingTestSet.txt')
    #训练集归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #生成NumPy数组,测试集
    inArr = array([precentTats, ffMiles, iceCream])
    #测试集归一化
    norminArr = (inArr - minVals) / ranges
    #返回分类结果
    classifierResult = classify0(norminArr, normMat, datingLabels, 3)
    #打印结果
    print("你可能%s这个人" % (resultList[classifierResult-1]))


if __name__ == '__main__':
    #
    #demo01()
    #
    #demo02()
    #归一化
    #demo03()
    #
    #datingClassTest()
    #
    classifyPerson()
