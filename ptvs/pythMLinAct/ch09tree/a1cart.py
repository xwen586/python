#!/usr/bin/env python3
# a1cart.py
""" CART算法的实现代码
"""
from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        #python3中map的返回值变了
        #fltLine = map(float,curLine) #map all elements to float()
        fltLine = list(map(float, curLine)) # 将每行映射成浮点数
        dataMat.append(fltLine)
    return dataMat


''' -------------- 将CART算法用于回归 --------------
伪代码大致如下：
找到最佳的待切分特征：
    如果该节点不能再分，将该节点存为叶节点
    执行二元切分
    在右子树调用createTree()方法
    在左子树调用createTree()方法
binSplitDataSet函数测试：
>>> from imp import reload
>>> import ch09tree.a1cart as rt
>>> testMat=mat(eye(4))
>>> mat0,mat1=rt.binSplitDataSet(testMat,1,0.5) #指定列1元素，按0.5拆分
mat0为行1值，mat1为行0,2,3值
>>> myDat=rt.loadDataSet(r'./ch09tree/data/ex00.txt')
>>> myMat = mat(myDat)
>>> rt.createTree(myMat)
'''
#拆分数据集函数，二元拆分法    
#@dataSet：待拆分的数据集
#@feature：作为拆分点的特征索引
#@value：特征的某一取值作为分割值
def binSplitDataSet(dataSet, feature, value):
    #采用条件过滤的方法,获取数据集每个样本目标特征的取值大于value的样本存入mat0
    #mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]
    #mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1

#叶节点生成函数:回归树
def regLeaf(dataSet):#returns the value used for each leaf
    return mean(dataSet[:,-1])#数据集列表最后一列特征值的均值作为叶节点返回

#误差计算函数:回归误差
def regErr(dataSet): ##计算数据集最后一列特征值的方差*数据集样本数，得到总方差返回
    return var(dataSet[:,-1]) * shape(dataSet)[0]

'''---切分函数---
目标是找到数据集切分的最佳位置。
它遍历所有的特征及其可能的取值来找到使误差最小化的切分阈值。
'''
#@dataSet：数据集
#@leafType：生成叶节点的类型，默认为回归树类型
#@errType：计算误差的类型，默认为总方差类型
#@ops：用户指定的参数，默认tolS=1.0，tolN=4
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    #用于控制函数的停止时机
    tolS = ops[0]; tolN = ops[1] #tolS容忍误差下降值1，最少切分样本数4
    #if all the target variables are the same value: quit and return value
    # 统计不同剩余特征值的数目,如果为1退出
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet) #该误差S将用于与新切分误差进行对比,来检查新切分能否降低误差。
    #初始化最小误差；最佳切分特征索引；最佳切分特征值
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):#遍历数据集所有的特征，除最后一列目标变量值
        #for splitVal in set(dataSet[:,featIndex]): Python3报错，matrix类型不能被hash
        for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS: #保留最小误差及对应的特征及特征值
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS: #如果切分后比切分前误差下降值未达到tolS
        return None, leafType(dataSet) #exit cond 2
    #检查最佳特征及特征值是否满足不切分条件
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #exit cond 3
        return None, leafType(dataSet)
    return bestIndex,bestValue#returns the best feature to split on
                              #and the value used for that split


#创建树函数，递归函数
#@dataSet：数据集
#@leafType：生成叶节点的类型 1 回归树：叶节点为常数值 2 模型树：叶节点为线性模型
#@errType：计算误差的类型
# 1-回归错误类型：总方差=均方差*样本数
# 2-模型错误类型：预测误差(y-yHat)平方的累加和
#@ops：用户指定的参数，包含tolS：容忍误差的降低程度 tolN：切分的最少样本数
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    #选取最佳分割特征和特征值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#choose the best split
    #如果特征为none，直接返回叶节点值
    if feat == None: return val #if the splitting hit a stop condition return val
    retTree = {}  #树的类型是字典类型
    retTree['spInd'] = feat  # 添加最佳特征属性
    retTree['spVal'] = val   # 添加最佳切分特征值属性
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree  

#
def demo01():
    myDat=loadDataSet(r'./ch09tree/data/ex00.txt')
    myMat = mat(myDat)
    plt.plot(myMat[:,0],myMat[:,1], 'ro')
    plt.show()
    rt = createTree(myMat)
    print('样本数据：ex00.txt')
    print(rt)
    #
    myDat1=loadDataSet(r'./ch09tree/data/ex0.txt')
    myMat1 = mat(myDat1)
    plt.plot(myMat1[:,0],myMat1[:,1], 'bo')
    plt.show()
    rt1 = createTree(myMat1)
    print('样本数据：ex0.txt')
    print(rt1)

#
def demo02():
    #树构建算法其实对输入的参数和敏感
    myDat2=loadDataSet(r'./ch09tree/data/ex2.txt')
    myMat2 = mat(myDat2)
    plt.plot(myMat2[:,0],myMat2[:,1], 'go')
    plt.show()
    print('样本数据：ex2.txt，默认ops=(1,4)')
    rt2 = createTree(myMat2) #默认ops=(1,4)
    print(rt2)
    print('调整ops=(10000,4)，停止条件tolS对误差的数量级十分敏感')
    rt3 = createTree(myMat2, ops=(10000,4))
    print(rt3)



''' -------------- 树剪枝 --------------
如果树节点过多，则该模型可能对数据过拟合，通过降低决策树的
复杂度来避免过拟合的过程称为剪枝。
'''
#判断输入是否为一棵树
def isTree(obj):
    return (type(obj).__name__=='dict')

#返回树的平均值
def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

# 后剪枝
#@tree:树字典    
#@testData:用于剪枝的测试集
def prune(tree, testData):
    # 确认测试数据集非空
    if shape(testData)[0] == 0: return getMean(tree) #if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):#if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #剪枝后判断是否还是有子树
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        #判断是否merge
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        #如果合并后误差变小
        if errorMerge < errorNoMerge: 
            print("merging")
            return treeMean
        else: return tree
    else: return tree

# 剪枝    
def demo03():
    # 创建所有可能中最大的树，如下命令：
    myDat2=loadDataSet(r'./ch09tree/data/ex2.txt')
    myMat2 = mat(myDat2)
    plt.plot(myMat2[:,0],myMat2[:,1], 'go')
    plt.show()
    print('样本数据：ex2.txt，创建最大树，ops=(0,1)')
    myTree = createTree(myMat2, ops=(0,1))
    print(myTree)
    # 
    myDat2t=loadDataSet(r'./ch09tree/data/ex2test.txt')
    myMat2t = mat(myDat2t)
    plt.plot(myMat2t[:,0], myMat2t[:,1], 'ro')
    plt.show()
    print('样本数据：ex2.txt，最大的树示例，ops=(0,1)')
    #rt2 = createTree(myMat2t) #默认ops=(1,4)
    #print(rt2)
    # 执行剪枝过程
    prune(myTree, myMat2t)


''' -------------- 模型树 --------------
采用树结构对数据建模，除了将叶节点设定为常数，也可将其设为分段线性函数。
>>> reload(rt)
>>> myMat2 = mat(rt.loadDataSet(r'./ch09tree/data/exp2.txt'))
>>> rt.createTree(myMat2, rt.modelLeaf, rt.modelErr, (1,10))
'''
#模型树
def linearSolve(dataSet):   #将数据集格式化为X Y
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]
    xTx = X.T*X
    if linalg.det(xTx) == 0.0: #X Y用于简单线性回归，需要判断矩阵可逆
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):#不需要切分时生成模型树叶节点
    ws,X,Y = linearSolve(dataSet)
    return ws #返回回归系数

def modelErr(dataSet):#用来计算误差找到最佳切分
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))

# 模型树
def demo04():
    myDat2 = loadDataSet(r'./ch09tree/data/exp2.txt')
    myMat2 = mat(myDat2)
    plt.plot(myMat2[:,0], myMat2[:,1], 'ro')
    plt.show()
    print('模型树')
    myTree = createTree(myMat2, modelLeaf, modelErr, (1,10))
    print(myTree)


''' --------------示例：树回归与标准回归的比较--------------
>>>trainMat=mat(rt.loadDataSet(r'./ch09tree/data/bikeSpeedVsIq_train.txt'))
>>>testMat=mat(rt.loadDataSet(r'./ch09tree/data/bikeSpeedVsIq_test.txt'))
>>>myTree=rt.createTree(trainMat, ops=(1,20))
>>>yHat = rt.createForeCast(myTree, testMat[:,0])
>>>corrcoef(yHat, testMat[:,1],rowvar=0)[0,1]
>>>myTree=rt.createTree(trainMat, rt.modelLeaf,rt.modelErr,(1,20))
>>>yHat = rt.createForeCast(myTree, testMat[:,0],rt.modelTreeEval)
>>>corrcoef(yHat, testMat[:,1],rowvar=0)[0,1]
>>>ws,X,Y=rt.linearSolve(trainMat)
>>>for i in range(shape(testMat)[0]): yHat[i]=testMat[i,0]*ws[1,0]+ws[0,0]
>>>corrcoef(yHat, testMat[:,1],rowvar=0)[0,1]
'''
#用树回归进行预测
#1-回归树
def regTreeEval(model, inDat):
    return float(model)

#2-模型树
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

#对于输入的单个数据点，treeForeCast返回一个预测值。
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)#指定树类型
    if inData[tree['spInd']] > tree['spVal']: #有左子树 递归进入子树
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:#不存在子树 返回叶节点
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)

#对数据进行树结构建模        
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat

# 模型树
def demo05():
    trainMat = mat(loadDataSet(r'./ch09tree/data/bikeSpeedVsIq_train.txt'))
    testMat = mat(loadDataSet(r'./ch09tree/data/bikeSpeedVsIq_test.txt'))
    myTree = createTree(trainMat, ops=(1,20))
    yHat = createForeCast(myTree, testMat[:,0])
    print(corrcoef(yHat, testMat[:,1],rowvar=0)[0,1])
    myTree=createTree(trainMat, modelLeaf, modelErr,(1,20))
    yHat = createForeCast(myTree, testMat[:,0], modelTreeEval)
    print(corrcoef(yHat, testMat[:,1],rowvar=0)[0,1])
    ws,X,Y=linearSolve(trainMat)
    print(ws)
    for i in range(shape(testMat)[0]):
       yHat[i]=testMat[i,0]*ws[1,0]+ws[0,0]
    corrcoef(yHat, testMat[:,1],rowvar=0)[0,1]


if __name__ == '__main__':
	#
    #demo01()
	#
    #demo02()
    #
    #demo03()
    #
    #demo04()
    #
    demo05()
