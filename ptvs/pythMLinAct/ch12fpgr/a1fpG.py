#!/usr/bin/env python3
# a1fpG.py
""" 使用FP-growth算法来高效发现频繁项集
"""

# FP树的节点类
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur  #计数器
        self.nodeLink = None
        self.parent = parentNode  #父变量 needs to be updated
        self.children = {} 
    
    def inc(self, numOccur):
        self.count += numOccur
        
    def disp(self, ind=1):
        print('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)

# 创建树
# dataSet，字典类型
def createTree(dataSet, minSup=1): #create FP-tree from dataset but don't mine
    headerTable = {}  # 头指针表,dict类型
    #go over dataSet twice
    ##第一次遍历数据集， 记录每个数据项的支持度
    for trans in dataSet: #first pass counts frequency of occurance
        #print(type(trans),trans) # trans为frozenset类型：<class 'frozenset'>
        for item in trans:
            #print(type(item), item)  #item为字符串类型 <class 'str'>
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    #根据最小支持度过滤
    for k in list(headerTable.keys()):  #remove items not meeting minSup
        if headerTable[k] < minSup: # 移除不满足最小支持度的元素项
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    # print('freqItemSet: ',freqItemSet)
    # 如果没有元素项满足要求，则退出
    if len(freqItemSet) == 0: return None, None  #if no items meet min support -->get out
    for k in headerTable:
        headerTable[k] = [headerTable[k], None] #reformat headerTable to use Node link 
    #print('headerTable: ',headerTable)
    #创建树，创建只包含空集合φ的根节点
    retTree = treeNode('Null Set', 1, None) #create tree
    ##第二次遍历数据集，创建FP树
    for tranSet, count in dataSet.items():  #go through dataset 2nd time
        localD = {} #根据最小支持度处理一条训练样本，key:样本中的一个样例，value:该样例的的全局支持度
        #根据全局频率对每个事务中的元素进行排序
        for item in tranSet:  #put transaction items in order
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            #根据全局频繁项对每个事务中的数据进行排序,等价于 order by p[1] desc, p[0] desc
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)#populate tree with ordered freq itemset
    return retTree, headerTable #return tree and header table

# 
def updateTree(items, inTree, headerTable, count):
    #第一个元素项是否作为子节点存在
    if items[0] in inTree.children:#check if orderedItems[0] in retTree.children
        inTree.children[items[0]].inc(count) #incrament count
    else:   #add items[0] to inTree.children
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        #更新头指针表,以指向新的节点。
        if headerTable[items[0]][1] == None: #update header table 
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    #剩下的元素项迭代
    if len(items) > 1:#call updateTree() with remaining ordered items
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

#确保节点链接指向树中该元素项的每一个实例。
def updateHeader(nodeToTest, targetNode):   #this version does not use recursion
    while (nodeToTest.nodeLink != None):    #Do not use recursion to traverse a linked list!
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


'''-------------- FP树生成示例 --------------
'''
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

# 生成数据集
def createInitSet(dataSet):
    retDict = {} # dict类型
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

def demo01():
    simpDat = loadSimpDat()  #加载数据
    initSet = createInitSet(simpDat) #数据进行格式化处理，生成字典类型
    myFPtree, myHeaderTab = createTree(initSet, 3)
    print('Head:', list(myHeaderTab.keys()))
    myFPtree.disp()


'''-------------- 从一棵FP树中挖掘频繁项集 --------------
首先从单元素项集合开始，然后在此基础上逐步构建更大的集合。
从FP树中抽取频繁项集的三个基本步骤如下：
(1)从FP树中获得条件模式基；
(2)利用条件模式基，构建一个条件FP树；
(3)迭代重复步骤(1)步骤（2），直到树包含一个元素项为止。
https://www.cnblogs.com/bigmonkey/p/7491405.html
'''
#上溯FP树，并收集所有遇到的元素项的名称
def ascendTree(leafNode, prefixPath): #ascends from leaf node to root
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath) #迭代上溯整棵树


#前缀路径发现函数,发现以给定元素项结尾的所有路径    
def findPrefixPath(basePat, treeNode): #treeNode comes from header table
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1: 
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

#上溯FP树
def demo02():
    simpDat = loadSimpDat()  #加载数据
    initSet = createInitSet(simpDat) #数据进行格式化处理，生成字典类型
    myFPtree, myHeaderTab = createTree(initSet, 3)
    print(findPrefixPath('x', myHeaderTab['x'][1]))
    print(findPrefixPath('z', myHeaderTab['z'][1]))
    print(findPrefixPath('r', myHeaderTab['r'][1]))


#递归查找频繁项集
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    #bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]#(sort header table)
    # order by minSup asc, value asc
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: (p[1][0],p[0]))]
    for basePat in bigL:  #start from bottom of header table
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        #print('finalFrequent Item: ', newFreqSet)    #append to set
        freqItemList.append(newFreqSet)
        # 通过条件模式基找到的频繁项集
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        #print('condPattBases :',basePat, condPattBases)
        #2. construct cond FP-tree from cond. pattern base
        myCondTree, myHead = createTree(condPattBases, minSup)
        #print 'head from conditional tree: ', myHead
        if myHead != None: #3. mine cond. FP-tree
            print('conditional tree for: ',newFreqSet)
            myCondTree.disp()            
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

def demo03():
    simpDat = loadSimpDat()  #加载数据
    initSet = createInitSet(simpDat) #数据进行格式化处理，生成字典类型
    myFPtree, myHeaderTab = createTree(initSet, 3)
    myFPtree.disp()
    condPats = findPrefixPath('z', myHeaderTab['z'][1])
    print('z', condPats)
    condPats = findPrefixPath('x', myHeaderTab['x'][1])
    print('x', condPats)
    condPats = findPrefixPath('y', myHeaderTab['y'][1])
    print('y', condPats)
    condPats = findPrefixPath('t', myHeaderTab['t'][1])
    print('t', condPats)
    condPats = findPrefixPath('s', myHeaderTab['s'][1])
    print('s', condPats)
    condPats = findPrefixPath('r', myHeaderTab['r'][1])
    print('r', condPats)
    minSup = 3
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)



if __name__ == '__main__':
	#
    #demo01()
	#上溯FP树
    #demo02()
	#
    demo03()
