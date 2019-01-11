#!/usr/bin/env python3
# a1logReg.py
""" Logistic回归 (最优化算法)
梯度算子总是指向函数值增长最快的方向。
参见 https://www.cnblogs.com/zy230530/p/6875145.html 文中描述

"""
from numpy import *
import matplotlib.pyplot as plt


''' ------------------ 基于最优化方法的最佳回归系数确定 ------------------
1）梯度上升的最优化方法，求得数据集的最佳参数
2）绘制梯度上升法产生的决策边界图
3）随机梯度上升算法
梯度上升算法用来求函数的最大值，而梯度下降算法用来求函数的最小值。
'''
# 读取文件testSet.txt中数据：前两列值分别是X1和X2，第3列是类别标签
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open(r'./ch05logi/testSet.txt')
    for line in fr.readlines():  # 并逐行读取
        lineArr = line.strip().split()   # 每行取值
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #拼接矩阵，增加值为1.0的第一列
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

# Sigmoid函数，又叫Logistic函数
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

''' 梯度上升算法，伪代码如下：
>每个回归系数初始化为1
>重复R次：
>    计算整个数据集的梯度
>    使用alpha * gradient 更新回归系数的向量
>    返回回归系数
回归系数进行更新的公式为：w=w+alpha*gradient，其中gradient是对参数w求偏导数。
给定一个数据两个维度的值，该函数能够预测其属于类别1的概率。
假设这个函数的模样如下：
 h(x) =sigmoid(z)
 z = w0 +w1*X1+w2*X2
问题转化成了，根据现有的样本数据，找出最佳的参数w(w0，w1，w2)的值
'''
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    labelMat = mat(classLabels).transpose() #变为矩阵后向量转置
    m, n = shape(dataMatrix) # 矩阵规格：100, 3
    alpha = 0.001  # 向目标移动的步长
    maxCycles = 500  # 迭代次数
    weights = ones((n,1))  # 本例n=3，即 x0, x1, x2
    # 迭代找出最佳的weights
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid( dataMatrix*weights )     #matrix mult
        error = (labelMat - h)   #真实类别和预测类别的差值 vector subtraction
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    return weights

''' 测试样例 Demo1()
'''
def demo01():
    print( '-- demo01 --' )
    dataArr, labelMat=loadDataSet()
    weights = gradAscent(dataArr, labelMat)
    print(weights)


''' 分析数据：画出决策边界
 画出数据集和Logistic回归最佳拟合直线的函数
'''
def plotBestFit(weights):
    dataMat, labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def demo02():
    print( '-- demo02 --' )
    dataArr, labelMat=loadDataSet()
    weights = gradAscent(dataArr, labelMat)
    print( weights.getA() )
    plotBestFit(weights.getA())


''' ------------------ 训练算法：随机梯度上升 ------------------ 
前边的梯度上升法gradAscent() 每次更新回归系数都需要遍历整个数据集，当样本数量较小时，该方法尚可。
当样本数据集非常大且特征非常多时，那么随机梯度下降法的计算复杂度就会特别高。
改进方法是一次仅用一个样本点来更新回归系数，该方法称为随机梯度上升算法，伪代码：
>所有回归系数初始化为1
>对数据集中每个样本
>     计算该样本的梯度
>     使用alpha x gradient 新回归系数值
>返回回归系数值
'''
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        #h = sigmoid(dataMatrix[i]*weights)
        error = classLabels[i] - h  #计算当前样本的残差(代替梯度)
        weights = weights + alpha * error * dataMatrix[i] #更新权值参数
    return weights

# 改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))  #python3.x  range返回的是range对象，不返回数组对象
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not 
            randIndex = int(random.uniform(0, len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            #h = sigmoid(dataMatrix[randIndex]*weights)
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

#
def demo03():
    print( '-- demo03 --' )
    dataArr, labelMat=loadDataSet()
    weights = stocGradAscent0(array(dataArr), labelMat)
    print( weights )
    plotBestFit(weights)
    weights = stocGradAscent1(array(dataArr), labelMat)
    print( weights )
    plotBestFit(weights)

''' ------------------ 示例：从疝气病症预测病马的死亡率 ------------------ 
使用Logistic回归来预测，数据包含368个样本和28个特征。
# 准备数据：处理被据中的缺失值，处理之后保存成两个文件：
horseColicTest.txt  - 测试集
horseColicTraining.txt  -训练集
'''
# 分类决策函数：回归系数和特征向量作为输入来计算对应的Sigmoid值
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

# 回归预测算法
def colicTest():
    frTrain = open(r'./ch05logi/horseColicTraining.txt'); #打开训练数据集
    frTest = open(r'./ch05logi/horseColicTest.txt') #打开测试数据集
    trainingSet = []; trainingLabels = []
    # 逐行读取训练集数据，共299行，每行22个数据，最后一个为标签值。
    for line in frTrain.readlines():
        currLine = line.strip().split() # 对当前行进行特征分割  '\t'
        lineArr =[]  # 列表类型
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21])) #存入标签值
    #调用随机梯度上升法更新logistic回归的权值参数，可以自由设定迭代的次数
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 200)  # 原1000
    #统计测试数据集预测错误样本数量和样本总数
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines(): #遍历测试数据集的每个样本
        numTestVec += 1.0
        currLine = line.strip().split()  #'\t'
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        #利用分类预测函数对该样本进行预测，并与样本标签进行比较
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

#多次测试算法求取预测误差平均值
def multiTest():
    #设置测试次数为10次，并统计错误率总和
    numTests = 6; errorSum=0.0
    #每一次测试算法并统计错误率
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % \
        (numTests, errorSum/float(numTests)))


''' 主程序
'''
if __name__ == '__main__':
    # 最佳参数样例
    #demo01()
    # 图形显示
    #demo02()
    # 梯度上升优化并显示
    #demo03()
    # 示例
    multiTest()
