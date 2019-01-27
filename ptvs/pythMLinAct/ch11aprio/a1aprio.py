#!/usr/bin/env python3
# a1aprio.py
""" 使用Apriori算法进行关联分析
https://www.cnblogs.com/ybjourney/p/4847489.html
"""
from numpy import *

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


'''-------------- 使用Apriori算法来发现频繁集 -------------- 
'''
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
                
    C1.sort()
    return list(map(frozenset, C1))#use frozen set so we
                            #can use it as a key in a dict    

def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not can in ssCnt: #ssCnt.has_key(can):
                    ssCnt[can]=1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData

def demo01():
    dataSet = loadDataSet()
    C1 = createC1(dataSet)
    D=list(map(set,dataSet))
    L1,suppData0 = scanD(D, C1, 0.5)
    print("L1:", L1)
    print("suppData0:", suppData0)


'''-------------- 完整的Apriori算法 -------------- 
当集合中项的个数大于0时
  构建一个k个项组成的候选项集的列表
  检查数据以确认每个项集都是频繁的
  保留频繁项集并构建k+1项组成的候选项集的列表
'''
#构建多个商品对应的项集
def aprioriGen(Lk, k): #creates Ck
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): 
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: #if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j]) #set union
    return retList

def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)#scan DB to get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

def demo02():
    dataSet = loadDataSet()
    minSupport = 0.5  # 50%的支持度
    L,suppData = apriori(dataSet, minSupport)
    print("L:", L)
    print("suppData:", suppData)
    # 尝试一下70%的支持度
    L,suppData = apriori(dataSet, minSupport=0.7)
    print("L:", L)
    print("suppData:", suppData)


'''-------------- 从频繁项集中挖掘关联规则 -------------- 
找出关联规则
'''
#使用关联规则生成函数
def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    bigRuleList = []
    for i in range(1, len(L)):#only get the sets with two or more items
        for freqSet in L[i]: #遍历L中的每一个频繁项集并
            # 对每个频繁项集创建只包含单个元素集合的列表H1
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):#如果频繁项集元素数目超过2,那么会考虑对它做进一步的合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:#第一层时，后件数为1
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         

##生成候选规则集合：计算规则的可信度以及找到满足最小可信度要求的规则.集合右边一个元素
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    #针对项集中只有两个元素时，计算可信度
    prunedH = [] #返回一个满足最小可信度要求的规则列表 #create new list to return
    for conseq in H: #后件，遍历 H中的所有项集并计算它们的可信度值
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence
        if conf >= minConf: 
            print(freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))#添加到规则里，brl是前面通过检查的 bigRuleList
            prunedH.append(conseq)
    return prunedH

#生成更多的关联规则
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)): #频繁项集元素数目大于单个集合的元素数 #try further merging
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)#计算可信度
        #满足最小可信度要求的规则列表多于1,则递归来判断是否可以进一步组合这些规则
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

def demo03():
    dataSet = loadDataSet()
    L,suppData = apriori(dataSet, minSupport=0.5)
    rules = generateRules(L,suppData,minConf = 0.7)
    print(rules)


'''-------------- 示例：发现国会投票中的模式 -------------- 
1：需要安装votesmart，下载地址 https://github.com/votesmart/python-votesmart
2：需要到 http://votesmart.org/services_api.php 中注册，获取apikey，才能运行查询信息。

修改votesmart.py代码如下：
13行   import urllib, urllib2
改为：import urllib, urllib.error, urllib.request

209行  params = dict([(k,v) for (k,v) in params.iteritems() if v])
改为： params = dict([(k,v) for (k,v) in params.items() if v])

211行 urllib.urlencode(params) 改为 urllib.parse.urlencode(params)
213行 response = urllib2.urlopen(url).read()
改为：response = urllib.request.urlopen(url).read()

219行如下代码
except urllib2.HTTPError, e:
            raise VotesmartApiError(e)
except ValueError, e:
            raise VotesmartApiError('Invalid Response')
修改为：
except urllib.error.URLError as e:
            raise VotesmartApiError(e)
except ValueError as e:
            raise VotesmartApiError('Invalid Response')
即可。

安装：python setup.py install
'''
from time import sleep
from votesmart import votesmart  # 此模块需要单独下载
votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030' # 这里需要改换成自己的API key
# 收集美国国会议案中action ID的函数
def getActionIds():
    actionIdList = []; billTitleList = []
    fr = open('recent20bills.txt') 
    for line in fr.readlines():
        billNum = int(line.split('\t')[0]) # 得到了议案的ID
        try:
            billDetail = votesmart.votes.getBill(billNum) # 得到一个billDetail对象
            for action in billDetail.actions:  # 遍历议案中的所有行为
                if action.level == 'House' and \
                (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId) 
                    print('bill: %d has actionId: %d' % (billNum, actionId))
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:  # API调用时发生错误
            print("problem getting bill %d" % billNum)
        sleep(1) # 礼貌访问网站而做出些延迟，避免过度访问
    return actionIdList, billTitleList

#基于投票数据的事务列表填充函数
def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
    # 创建一个含义列表
    itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
    for billTitle in billTitleList:#fill up itemMeaning list
        itemMeaning.append('%s -- Nay' % billTitle) # 在议案标题后面添加Nay(反对)
        itemMeaning.append('%s -- Yea' % billTitle) # 在议案标题后添加Yea(同意)
    transDict = {} # 用于加入元素项 #list of items in each transaction (politician) 
    voteCount = 2
    for actionId in actionIdList: # 遍历getActionIds()返回的每一个actionId
        sleep(3) # 延迟访问，防止过于频繁的API调用 
        print('getting votes for actionId: %d' % actionId)
        try:
            # 获得某个特定的actionId的所有投票信息
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName): # 如果没有该政客的名字
                    transDict[vote.candidateName] = [] # 用该政客的名字作为键来填充transDict
                    # 获取该政客的政党信息
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except: 
            print("problem getting actionId: %d" % actionId)
        voteCount += 2
    return transDict, itemMeaning

# 国会投票模式，因没有注册获取apikey，程序不能完成
def demo04():
    actionIdList, billTitles = getActionIds()
    transDict, itemMeaning = getTransList(actionIdList, billTitles)
    dataSet = [transDict[key] for key in transDict.keys()]
    L, suppData = apriori(dataSet, minSupport=0.5)
    print(L)
    rules = generateRules(L, suppData)


'''-------------- 示例：发现毒蘑菇的相似特征 --------------

mushroom.dat中，第一列为特征，表示有毒(2)或者可食用(1)
'''
def readMushroom():
    mushDataSet = [line.split() for line in open(r'./data/mushroom.dat').readlines()]
    # 数据集上运行Apriori算法
    L, suppData=apriori(mushDataSet, minSupport = 0.3)
    print("频繁项集L:", len(L))
    # 含有毒特征值2的频繁项集
    for item in L[1]:
        if item.intersection('2'): print(item)
    # 对更大的项集来重复上述过程：
    for item in L[3]:
        if item.intersection('2'): print(item)



if __name__ == '__main__':
	#
    #demo01()
	#
    #demo02()
	#
    demo03()
