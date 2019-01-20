#!/usr/bin/env python3
# a1adabt.py
"""  AdaBoost算法

"""
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

'''建立一个简单的数据集
'''
def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

# 显示数据分布
def datashow():
    datMat, classLabels = loadSimpData()
    xcord0 = []
    ycord0 = []
    xcord1 = []
    ycord1 = []
    markers =[]
    colors =[]
    for i in range(len(classLabels)):
        if classLabels[i]==1.0:
            xcord1.append(datMat[i,0]), ycord1.append(datMat[i,1])
        else:
            xcord0.append(datMat[i,0]), ycord0.append(datMat[i,1])
    fig = plt.figure()
    ax = fig.add_subplot(111)       
    ax.scatter(xcord0,ycord0, marker='s', s=90)
    ax.scatter(xcord1,ycord1, marker='o', s=50, c='red')
    plt.title('decision stump test data')
    plt.show()


''' ------------ 单层决策树生成函数 ------------
第一个函数将用于测试是否有某个值小于或者大于我们正在测试的阈值。
第二个函数则更加复杂一些，它会在一个加权数据集中循环，并找到具有最低错误率的单层决策树。
'''
#单层决策树的阈值过滤函数，采用+1和-1作为类别
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):#just classify the data
    retArray = ones((shape(dataMatrix)[0],1)) # 初始值为1
    #阈值的模式，判断模式是否为小于（lt：litter）
    if threshIneq == 'lt': # 将小于某一阈值的特征归类为-1
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else: # 否则将大于某一阈值的特征归类为-1
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    
# 构建Stump(决策器），建立单层决策树(弱学习器)
def buildStump(dataArr, classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T #将数据集和标签列表转为矩阵形式
    m,n = shape(dataMatrix)
    #步长或区间总数 最优决策树信息 最优单层决策树预测结果
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf # init error sum, to +infinity
    for i in range(n):#loop over all dimensions
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            for inequal in ['lt', 'gt']: #go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr  #calc total error multiplied by D
                #print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % \
                #    (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

# 弱学习器演示
def demo01():
    datMat, classLabels=loadSimpData()
    D = mat(ones((5,1))/5)
    buildStump(datMat, classLabels, D)


''' ------------ 完整AdaBoost算法的实现 ------------
伪代码如下：
对每次迭代：
    利用buildStump()函数找到最佳的单层决策树
    将最佳单层决策树加入到单层决策树数组
    计算alpha
    计算新的权重向量D
    更新累计类别估计值
    如果错误率等于0.0,则退出循环
'''
#@dataArr：数据矩阵
#@classLabels:标签向量
#@numIt:迭代次数    
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)   #init D to all equal
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr,classLabels,D)#build Stump
        print("[%d]----- D:"% i, D.T)
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)                  #store Stump Params in Array
        print("classEst:", classEst.T)
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy
        D = multiply(D,exp(expon))    #Calc New D for next iteration
        D = D/D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        print("aggClassEst", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print("total error: ",errorRate)
        if errorRate == 0.0: break
    return weakClassArr, aggClassEst

# 训练的过程
def demo02():
    datMat, classLabels=loadSimpData()
    D = mat(ones((5,1))/5)
    adaBoostTrainDS(datMat, classLabels, 9)



''' ------------ 测试算法：基于AdaBoost的分类 ------------
>>>from imp import reload
>>>import ch07adaboost.a1adabt as ada
>>>datArr, labelArr=ada.loadSimpData()
>>>classifierArr, aggEst = ada.adaBoostTrainDS(datArr,labelArr,30)
>>>ada.adaClassify([0, 0],classifierArr[0])  # 此处书中有误
'''
#测试adaBoost分类函数
#@datToClass:测试数据点
#@classifierArr：构建好的最终分类器
def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, \
            classifierArr[i]['dim'], \
            classifierArr[i]['thresh'], \
            classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return sign(aggClassEst)

# 训练的过程
def demo03():
    datArr, labelArr=loadSimpData()
    classifierArr = adaBoostTrainDS(datArr,labelArr,30)
    adaClassify([0, 0], classifierArr[0]) # 此处源码有误
    adaClassify([[5, 5], [0,0]], classifierArr[0])



''' ------------ 实例：难数据集上应用adaBoost ------------
>>> reload(ada)
>>> datArr,labelArr = ada.loadDataSet(r'./ch07adaboost/data/horseColicTraining2.txt')
>>> classifierArray = ada.adaBoostTrainDS(datArr,labelArr,10)
>>> testArr,testLabelArr = ada.loadDataSet(r'./ch07adaboost/data/horseColicTest2.txt')
>>> prediction10 = ada.adaClassify(testArr,classifierArray[0])
>>> errArr=np.mat(np.ones((67,1)))
>>> errArr[prediction10!=np.mat(testLabelArr).T].sum()
'''
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def plotROC(predStrengths, classLabels):
    #import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is: ", ySum*xStep)

# 实例
def demo04():
    datArr,labelArr = loadDataSet(r'./ch07adaboost/data/horseColicTraining2.txt')
    classifierArray, aggClassEst = adaBoostTrainDS(datArr,labelArr,10)
    plotROC(aggClassEst.T, labelArr)


if __name__ == '__main__':
    #
    #datashow()
    # 弱学习器演示
	#demo01()
    # 训练的过程
    #demo02()
    #
    #demo03()
    #
    demo04()
