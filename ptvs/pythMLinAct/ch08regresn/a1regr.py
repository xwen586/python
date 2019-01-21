#!/usr/bin/env python3
# a1regr.py
""" 线性回归（linear regression）
□ 线性回归
□ 局部加权线性回归
□ 岭回归和逐步线性回归
□ 预测鲍鱼年龄和玩具售价
https://www.cnblogs.com/zy230530/p/6942458.html

"""
from numpy import *
import matplotlib.pyplot as plt

''' 读取文件数据集
>>> from imp import reload
>>> import ch08regresn.a1regr as reg
>>> xArr, yArr=reg.loadDataSet(r'./ch08regresn/data/ex0.txt')
>>> ws = reg.standRegres(xArr,yArr)
>>> xMat=mat(xArr)
>>> yMat=mat(yArr)
>>> yHat = xMat*ws
'''
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat


# 标准线性回归算法 w*=(XT*X)-1*XT*y
# ws = (X.T*X).I * (X.T*Y)    
def standRegres(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T  #将列表形式的数据转为numpy矩阵形式
    xTx = xMat.T*xMat  #求矩阵的内积：(X.T*X)
    # 判断xMat是否可逆
    if linalg.det(xTx) == 0.0: # det()计算矩阵的行列式
        print("This matrix is singular, cannot do inverse")
        return
    #如果可逆，根据公式计算回归系数
    ws = xTx.I * (xMat.T*yMat)    #可以用yHat=xMat*ws计算实际值y的预测值
    return ws  #返回回归系数 w0, w1

# 演示线性回归，并画图
def demo01():
    #读取的文件ex0.txt中，第一列是x0，常数1.0；第二列是输入变量x1，都存储在xArr中
    #第三列是y，存储在yArr。最终会得到y=ws[0]+ws[1]*X1
    xArr, yArr=loadDataSet(r'./ch08regresn/data/ex0.txt')
    ws = standRegres(xArr,yArr)
    xMat=mat(xArr) # 200*2 矩阵
    yMat=mat(yArr) # 真实的y值
    yHat = xMat*ws # 预测y值
    print('corrcoef:', corrcoef(yHat.T, yMat)) #相关系数,均方误差?
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制散点图
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
    xCopy=xMat.copy()
    #xCopy.sort(0)  #升序排列，防止数据点次序混乱
    yHat=xCopy*ws
    ax.plot(xCopy[:,1],yHat) # 折线图
    plt.show()


''' ------------ 局部加权线性回归 ------------
'''
# 局部加权线性回归(LWLR)
#每个测试点赋予权重系数
#@testPoint:测试点
#@xArr：样本数据矩阵
#@yArr：样本对应的原始值
#@k：用户定义的参数，决定权重的大小，默认1.0
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]  # 获取矩阵行数（样本数=200）
    weights = mat(eye((m))) #初始化权重矩阵为m*m的单位阵
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]     # 计算预测点与该样本的偏差，w0,w1
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))  # 高斯核函数求解权重
    xTx = xMat.T * (weights * xMat) #将权重矩阵应用到公式中
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat)) #计算回归系数
    return testPoint * ws

#测试集进行预测 用于为数据集中每个点调用lwlr()
def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m): #遍历每一个测试样本
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

# 演示局部加权线性回归，并画图
def demo02():
    xArr, yArr=loadDataSet(r'./ch08regresn/data/ex0.txt')
    #lwlr(xArr[0], xArr, yArr, 1.0)
    #lwlr(xArr[0], xArr, yArr, 0.01)
    #不同平滑系数k，拟合效果不同。1,0.1，0.003
    y1Hat = lwlrTest(xArr, xArr, yArr, 1.0)
    y2Hat = lwlrTest(xArr, xArr, yArr, 0.02)
    y3Hat = lwlrTest(xArr, xArr, yArr, 0.003)
    xMat=mat(xArr)
    srtInd = xMat[:,1].argsort(0)
    xSort=xMat[srtInd][:,0,:]
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.plot(xSort[:,1], yHat[srtInd])
    #ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
    fig, ax = plt.subplots(3,1, figsize=(9,6))
    ax[0].plot(xSort[:,1], y1Hat[srtInd], color='blue');
    ax[0].scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
    ax[1].plot(xSort[:,1], y2Hat[srtInd], color='blue');
    ax[1].scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
    ax[2].plot(xSort[:,1], y3Hat[srtInd], color='blue');
    ax[2].scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
    plt.tight_layout()  #自动调整子插图
    plt.show()


''' ------------ 示例：预测鲍鱼的年龄 ------------
鲍鱼的年龄可以通过鲍鱼壳的层数推算得到。岭回归，前向逐步回归
>>> reload(reg)
>>> abX,abY=reg.loadDataSet(r'./ch08regresn/data/abalone.txt')
>>> yHat01=reg.lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
>>> yHat1=reg.lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
>>> yHat10=reg.lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)
>>> reg.rssError(abY[0:99],yHat01.T)
>>> reg.rssError(abY[0:99],yHat1.T)
>>> reg.rssError(abY[0:99],yHat10.T)
>>> ridgeWeights=reg.ridgeTest(abX,abY)
'''
#计算平方误差的和
def rssError(yArr, yHatArr):
    #返回平方误差和
    return ((yArr-yHatArr)**2).sum()

#一、岭回归
#@xMat:样本数据
#@yMat：样本对应的原始值
#@lam：惩罚项系数lamda，默认值为0.2
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam #添加惩罚项，使矩阵xTx变换后可逆
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat) #计算回归系数
    return ws

#特征需要标准化处理，使所有特征具有相同重要性
def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #to eliminate X0 take mean off of Y
    #regularize X's
    xMeans = mean(xMat,0)   #calc mean then subtract it off
    xVar = var(xMat,0)      #calc variance of Xi then divide by it
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        #计算对应lamda回归系数，lamda以指数形式变换
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat


# 演示岭回归，并画图
def demo03():
    abX,abY=loadDataSet(r'./ch08regresn/data/abalone.txt')
    ridgeWeights=ridgeTest(abX,abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()


#二、前向逐步回归
def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

#前向逐步回归
#@eps：每次迭代需要调整的步长
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m,n=shape(xMat)
    #将每次迭代中得到的回归系数存入矩阵
    returnMat = zeros((numIt,n)) #testing code remove
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        #print(ws.T)
        lowestError = inf; 
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest #变化后计算相应预测值
                rssE = rssError(yMat.A,yTest.A) #保存最小的误差以及对应的回归系数
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

# 演示前向逐步回归，并画图
def demo04():
    xArr,yArr = loadDataSet(r'./ch08regresn/data/abalone.txt')
    #ridgeWeights=stageWise(xArr, yArr, 0.01, 200)
    ridgeWeights=stageWise(xArr, yArr,0.001, 5000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()


if __name__ == '__main__':
	#
    #demo01()
    #
    #demo02()
    #
    #demo03()
    #
    demo04()

