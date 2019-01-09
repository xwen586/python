#!/usr/bin/env python3
# a1tree.py
"""
决策树
决策树分类器与信息熵（香农熵）
构建决策树时，采用递归的方法将数据集转化为决策树。
"""
from math import log
import operator

''' 计算给定数据集的经验熵(香农熵)
数据格式参见 createDataSet() 函数
'''
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)	#计算数据集中实例的总数
    labelCounts = {}   # 保存每个标签出现次数
    # 创建一个数据字典，它的键值是最后一列的数值
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]  # 键值是最后一列的数值
        #如果当前键值不存在，则扩展字典并将当前键值加入字典
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1 # 每个键值都记录了当前类别出现的次数。
    # 计算香农熵：先计算概率，再求熵。
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries  # 计算类别出现的概率。
        shannonEnt -= prob * log(prob,2) # 计算香农熵 log base 2
    return shannonEnt

''' 创建测试数据集
'''
def createDataSet1():
    dataSet = [
               [1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers'] # 不浮出水面是否可以生存，是否有脚撲，属于鱼类
    #change to discrete values
    return dataSet, labels

def createDataSet2():
	dataSet = [[0, 0, 0, 0, 'no'],						#数据集
			[0, 0, 0, 1, 'no'],
			[0, 1, 0, 1, 'yes'],
			[0, 1, 1, 0, 'yes'],
			[0, 0, 0, 0, 'no'],
			[1, 0, 0, 0, 'no'],
			[1, 0, 0, 1, 'no'],
			[1, 1, 1, 1, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[2, 0, 1, 2, 'yes'],
			[2, 0, 1, 1, 'yes'],
			[2, 1, 0, 1, 'yes'],
			[2, 1, 0, 2, 'yes'],
			[2, 0, 0, 0, 'no']]
	labels = ['年龄', '有工作', '有自己的房子', '信贷情况']		#特征标签
	return dataSet, labels 							#返回数据集和分类属性

''' 测试香农熵
'''
def demo01():
    print( '-- demo01 --' )
    myDat, labels = createDataSet1()
    shan = calcShannonEnt(myDat)
    print( shan )
    # 测试熵的变化
    myDat[0][-1]='maybe'
    calcShannonEnt(myDat)
    print( '熵的变化：' + str(shan) )

def demo02():
    print( '-- demo02 --' )
    myDat, labels = createDataSet2()
    shan = calcShannonEnt(myDat)
    print( shan )

''' 按照给定特征划分数据集
将某个特征(axis)列下，等于value的数据行元素抽取出来。
	dataSet - 待划分的数据集
	axis - 划分数据集的特征
	value - 需要返回的特征的值
'''
def splitDataSet(dataSet, axis, value):
    retDataSet = []   # 声明一个新列表对象，即划分后的数据集
    for featVec in dataSet:  # 遍历数据集
        if featVec[axis] == value:   # 符合特征的数据抽取出来
            reducedFeatVec = featVec[:axis]     # 去掉axis特征 chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

''' 选择最好的数据集划分方式
'''
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      # 特征数量 the last column is used for the labels
    # 计算数据集的香农熵，保存最初的无序度量值，用于与划分完之后的数据集计算的嫡值进行比较。
    baseEntropy = calcShannonEnt(dataSet)  
    bestInfoGain = 0.0;  # 信息增益
    bestFeature = -1   # 最优特征的索引值
    for i in range(numFeatures):        # 遍历所有特征 iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #集合类型中，每个值互不相同 get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value) # i特征下值为value的子集
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        # 计算最好的信息增益
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

''' 划分数据集测试
'''
def demo11():
    print( '-- demo11 --' )
    myDat, labels = createDataSet1()
    bestFeature = chooseBestFeatureToSplit(myDat)
    print( bestFeature )

def demo12():
    print( '-- demo12 --' )
    myDat, labels = createDataSet2()
    bestFeature = chooseBestFeatureToSplit(myDat)
    print( bestFeature )

''' ------- 递归构建决策树 -------
'''
# 统计classList中出现次数最多的元素(类标签)
def majorityCnt(classList):
    classCount={}
    for vote in classList: #统计classList中每个元素出现的次数
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0] #返回classList中出现次数最多的元素

''' 创建决策树
输人参数：数据集和标签列表
'''
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    # 递归函数的第一个停止条件是所有的类标签完全相同，则直接返回该类标签
    if classList.count(classList[0]) == len(classList): 
        return classList[0]  #stop splitting when all of the classes are equal
    # 第二个停止条件是遍历完所有特征时，返回出现次数最多的类标签
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)  # 挑选出现次数最多的类别作为返回值。
    # 创建树
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}  # 用字典类型(dict)存储树的信息
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            

''' 划分数据集测试
'''
def demo21():
    print( '-- demo21 --' )
    myDat, labels = createDataSet1()
    myTree = createTree(myDat, labels)
    print( myTree )

def demo22():
    print( '-- demo22 --' )
    myDat, labels = createDataSet2()
    myTree = createTree(myDat, labels)
    print( myTree )


''' -------- 绘制树形图 ---------'''
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 获取决策树叶子结点的数目
def getNumLeafs(myTree):
    numLeafs = 0	#初始化叶子
    firstStr = next(iter(myTree))	#python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    secondDict = myTree[firstStr]	 #获取下一组字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':	 #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

# 获取决策树的层数
def getTreeDepth(myTree):
    maxDepth = 0								 #初始化决策树深度
    firstStr = next(iter(myTree))		 #python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    secondDict = myTree[firstStr]		 #获取下一个字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':		 #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth			#更新层数
    return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-") #定义箭头格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)		#设置中文字体
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )

# 标注有向边属性值
def plotMidText(cntrPt, parentPt, txtString):
	xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0] #计算标注位置					
	yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
	createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

# 绘制决策树
def plotTree(myTree, parentPt, nodeTxt):
	decisionNode = dict(boxstyle="sawtooth", fc="0.8")	 #设置结点格式
	leafNode = dict(boxstyle="round4", fc="0.8") #设置叶结点格式
	numLeafs = getNumLeafs(myTree)  #获取决策树叶结点数目，决定了树的宽度
	depth = getTreeDepth(myTree)	 #获取决策树层数
	firstStr = next(iter(myTree))		 #下一个字典     											
	cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)	#中心位置
	plotMidText(cntrPt, parentPt, nodeTxt) #标注有向边属性值
	plotNode(firstStr, cntrPt, parentPt, decisionNode)	 #绘制结点
	secondDict = myTree[firstStr]	 #下一个字典，也就是继续绘制子结点
	plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD #y偏移
	for key in secondDict.keys():								
		if type(secondDict[key]).__name__=='dict':		 #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
			plotTree(secondDict[key],cntrPt,str(key))    #不是叶结点，递归调用继续绘制
		else:	 #如果是叶结点，绘制叶结点，并标注有向边属性值 											
			plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
			plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
			plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
	plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

# 创建绘制面板
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white') #创建fig
    fig.clf()					 #清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops) #去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))	 #获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))	 #获取决策树层数
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;	 #x偏移
    plotTree(inTree, (0.5,1.0), '')	 #绘制决策树
    plt.show()								 #显示绘制结果

def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

''' 绘制决策树
'''
def demo31():
    print( '-- demo31 --' )
    myTree = retrieveTree(1)
    # myTree = createTree(myDat, labels)
    leafs = getNumLeafs(myTree)
    print('leafs: ' + str(leafs)  )
    depth = getTreeDepth(myTree)
    print('depth: ' + str(leafs)  )
    createPlot(myTree)
    #
    print( '-- demo31 tree--' )
    myDat, labels = createDataSet1()
    myTree = createTree(myDat, labels)
    createPlot(myTree)

''' -------- 测试算法：使用决策树执行分类 -------- 
依靠训练数据构造了决策树之后，可以将它用于实际数据的分类。
# 使用决策树的分类器函数
测试：
>>>from imp import reload
>>>import ch03dt.a1tree
>>>reload(ch03dt.a1tree)
>>>myDat,labels=ch03dt.a1tree.createDataSet1()
>>>myTree=ch03dt.a1tree.retrieveTree (0)
>>>ch03dt.a1tree.classify(myTree,labels,[1,0])
'''
def classify(inputTree, featLabels, testVec):
    #firstStr = inputTree.keys()[0]  # 获取决策树结点 python2的
    firstStr = next(iter(inputTree))  # 迭代函数iter()与next()合用，获取第一个节点
    secondDict = inputTree[firstStr]  # 下一个字典
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

''' 存储决策树
>>> reload(ch03dt.a1tree)
>>> ch03dt.a1tree.storeTree(myTree,'classifierStorage.txt')
'''
def storeTree(inputTree, filename):
    import pickle
    # with open(filename, 'wb') as fw:
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

''' 读取决策树
>>> ch03dt.a1tree.grabTree('classifierStorage.txt')
'''
def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


''' 示例：使用决策树预测隐形眼镜类型
>>> lensesTree = ch03dt.a1tree.createTree(lenses, lensesLabels)
'''
def test():
    fr = open(r'./ch03dt/lenses.txt')
    lenses=[inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels=['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    createPlot(lensesTree)


''' 主程序
'''
if __name__ == '__main__':
    # 
    demo01()
    demo02()
    # 
    demo11()
    demo12()
    # 创建决策树
    demo21()
    demo22()
    # 
    demo31()
    #
    test()
