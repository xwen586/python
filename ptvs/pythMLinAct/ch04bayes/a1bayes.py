#!/usr/bin/env python3
# a1bayes.py
""" 基于概率论的分类方法：朴素贝叶斯
是用于文档分类的常用算法。
从文本中获取特征，特征是来自文本的词条(token)，即单词。然后将每一个文本片段表示为一个词条向量，其中值为1表示词条出现在文档中，0表示词条未出现。
侮辱类和非侮辱类，使用1和0分别表示。
"""
from numpy import *

'''------------------ 准备数据：从文本中构建词向量 ------------------
如何将一组单词转换为一组数字:
多个文档(postingList) ；标注各文档侮辱类标记(classVec)
生成词汇表 createVocabList()  是将文档的词汇汇总
词表到向量的转换
>>> from imp import reload
>>> import a1bayes as bayes
>>> listOPosts,listClasses = bayes.loadDataSet()
>>> myVocabList = bayes.createVocabList(listOPosts)
>>> myVocabList
>>> bayes.setOfWords2Vec(myVocabList, listOPosts[0])
>>> bayes.setOfWords2Vec(myVocabList, listOPosts[3])
'''
# 创建实验样本
def loadDataSet():
    # 来自斑点犬爱好者留言，词条切分后的文档集合，每一行代表一个文档
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 类别标签的集合
    classVec = [0,1,0,1,0,1]    # 1代表侮辱性文字，0代表正常言论 1 is abusive, 0 not
    return postingList,classVec

# 生成词汇表：使用set数据类型，创建不重复词的列表
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        #将文档列表转为集合的形式，保证每个词条的唯一性
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)  #再将集合转化为列表

# 词表到向量的转换函数
# 将词汇表转化为词条向量：inputSet-输入文档，vocabList-词条列表
def setOfWords2Vec(vocabList, inputSet):
    #新建一个长度为vocabSet的列表，并且各维度元素初始化为0
    returnVec = [0]*len(vocabList)
    for word in inputSet:  #遍历文档中的每一个单词
        if word in vocabList:   # 单词在词条列表中出现
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec  # 返回inputSet转化后的词条向量


'''------------------ 训练算法：从词向量计算概率 ------------------
在给定文档类别条件下词汇表中单词的出现概率
如何使用这些数字计算概率，如下命令外，参见demo01()
>>> from numpy import *
>>> reload(bayes)
>>> listOPosts, listClasses = bayes.loadDataSet()
>>> myVocabList = bayes.createVocabList(listOPosts)
>>> myVocabList
>>> trainMat=[]
>>> for postinDoc in listOPosts: trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))
>>> p0V,p1V,pAb = bayes.trainNB0(trainMat,listClasses)
>>> pAb   #侮辱类的概率pAb为0.5
'''
#训练算法，从词向量计算概率p(w0|ci)...及p(ci)
#@trainMatrix：由每篇文档的词条向量组成的文档矩阵
#@trainCategory：每篇文档的类标签组成的向量
def trainNB0(trainMatrix, trainCategory):
    #获取文档矩阵中文档的数目
    numTrainDocs = len(trainMatrix)
    #获取词条向量的长度
    numWords = len(trainMatrix[0])
    #所有文档中属于类1所占的比例p(c=1):  3/6=0.5
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #创建一个长度为词条向量等长的列表。在trainNB1中会算法改进
    p0Num = zeros(numWords);  p1Num = zeros(numWords)
    p0Denom = 0.0;  p1Denom = 0.0
    #遍历每一篇文档的词条向量
    for i in range(numTrainDocs):
        #如果该词条向量对应的标签为1
        if trainCategory[i] == 1:  # 1代表侮辱性
            #统计所有类别为1的词条向量中各个词条出现的次数
            p1Num += trainMatrix[i]
            #统计类别为1的词条向量中出现的所有词条的总数
            #即统计类1所有文档中出现单词的数目
            p1Denom += sum(trainMatrix[i])
        else:
            #统计所有类别为0的词条向量中各个词条出现的次数
            p0Num += trainMatrix[i]
            #统计类别为0的词条向量中出现的所有词条的总数
            #即统计类0所有文档中出现单词的数目
            p0Denom += sum(trainMatrix[i])
    #利用NumPy数组计算p(wi|c1)
    p1Vect = p1Num/p1Denom  #为避免下溢出问题，后面会改为log()
    #数组计算p(wi|c0)
    p0Vect = p0Num/p0Denom  #为避免下溢出问题，后面会改为log()
    return p0Vect, p1Vect, pAbusive

'''  训练算法改进
1）第4行和第5行修改为：
#p0Num = ones(numWords); p1Num = ones(numWords)
#p0Denom = 2.0; p1Denom = 2.0
2）解决下溢出问题
'''
def trainNB1(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # 算法改进
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones() 
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 1代表侮辱性
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:  # 正常言论
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 下溢出问题改进，通过求对数可以避免下溢出或者浮点数舍入导致的错误。
    p1Vect = log(p1Num/p1Denom)          #change to log()
    p0Vect = log(p0Num/p0Denom)          #change to log()
    return p0Vect, p1Vect, pAbusive

''' 使用训练函数trainNB() 样例
'''
def demo01():
    print( '-- demo01 --' )
    listOPosts, listClasses = loadDataSet()  # 样本文档和带侮辱分类标记的数据生成
    myVocabList = createVocabList(listOPosts)  # 将文档的词汇汇总，生成词汇表
    print( myVocabList )
    trainMat=[]  # 转换为数字向量的矩阵表
    for postinDoc in listOPosts:  # 遍历样本文档
        vec = setOfWords2Vec(myVocabList, postinDoc) # 样本文档对词汇表的向量转换
        trainMat.append(vec)  # 追加
    #使用优化后的分类概率：p0V-正常概率; p1V-侮辱性言论概率; pAb-标记中的侮辱性言论概率
    p0V, p1V, pAb = trainNB1(trainMat, listClasses) 
    # p0V, p1V, pAb = trainNB0(trainMat, listClasses) 
    print("p0V-正常言论概率")
    print(p0V)
    print("p1V-侮辱性言论概率")
    print(p1V)


''' ------------------ 测试算法：根据现实情况修改分类器 ------------------
>>> reload(bayes)
>>> bayes.testingNB()
'''
# 朴素贝叶斯分类函数
#@vec2Classify:待测试分类的词条向量
#@p0Vec:类别0所有文档中各个词条出现的频数p(wi|c0)
#@p1Vec:类别1所有文档中各个词条出现的频数p(wi|c1)
#@pClass1:类别为1的文档占文档总数比例
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #根据朴素贝叶斯分类函数分别计算待分类文档属于类1和类0的概率
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

#分类测试函数
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts) #统计生成词条列表
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    #调用训练函数，得到相应概率值
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    #测试文档
    testEntry = ['love', 'my', 'dalmation']
    #将测试文档转为词条向量，并转为数组的形式
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    #利用贝叶斯分类函数对测试文档进行分类并打印
    print(testEntry,'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))
    #第二个测试文档
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))


''' ------------------ 准备数据：文档词袋模型 ------------------
前边的代码，使用词集模型，即以该词是否出现来计算
词袋中，统计每个单词出现次数
'''
#朴素贝叶斯词袋模型.   基于词袋模型的朴素贝叶斯代码
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


'''
------------------ 示例：使用朴素贝叶斯过滤垃圾邮件 ------------------
1、准备数据：切分文本
>>> reload(bayes)
>>> mySent='This book is the best book on Python or M.L. I have ever laid ➥ eyes upon.'
>>> mySent.split()
>>> import re
>>> regEx = re.compile('\\W*')
>>> listOfTokens = regEx.split(mySent)
>>> [tok.lower() for tok in listOfTokens if len(tok) > 0]
>>> emailText = open('email/ham/23.txt').read()
>>> listOfTokens = regEx.split(emailText)
2、使用朴素贝叶斯进行交叉验证
>>> bayes.spamTest()
'''
# 切分文本
#1 对长字符串进行分割，分隔符为除单词和数字之外的任意符号串
#2 将分割后的字符串中所有的大些字母变成小写lower(),并且只
#保留单词长度大于3的单词
def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 

def pwd():
	import os
	path = os.getcwd()
	print('PATH:' + path)
	print( os.path.exists('ch04bayes/email/spam') )


#贝叶斯算法实例：过滤垃圾邮件
def spamTest():
    #新建三个列表
    docList=[]; classList=[]; fullTest=[]
    #i 由1到26
    for i in range(1,26):
        #打开并读取指定目录下的本文中的长字符串，并进行处理返回
        wordList = textParse(open(r'./ch04bayes/email/spam/%d.txt' % i).read())
        #将得到的字符串列表添加到docList
        docList.append(wordList)
        #将字符串列表中的元素添加到fullTest
        fullTest.extend(wordList)
        #类列表添加标签1
        classList.append(1)
        #打开并取得另外一个类别为0的文件，然后进行处理
        wordList = textParse(open(r'./ch04bayes/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(0)
    #将所有邮件中出现的字符串构建成字符串列表
    vocabList = createVocabList(docList)
    #构建一个大小为50的整数列表和一个空列表
    trainingSet = list(range(50))  #python3.x range返回的是range对象，不返回数组对象
    testSet=[]
    #随机选取1~50中的10个数，作为索引，构建测试集
    for i in range(10):
        #随机选取1~50中的一个整型数
        randIndex = int(random.uniform(0,len(trainingSet)))
        #将选出的数的列表索引值添加到testSet列表中
        testSet.append(trainingSet[randIndex])
        #从整数列表中删除选出的数，防止下次再次选出
        #同时将剩下的作为训练集
        del(trainingSet[randIndex])

    #新建两个列表
    trainMat = [];  trainClasses = []
    #遍历训练集中的吗每个字符串列表
    for docIndex in trainingSet:
        #将字符串列表转为词条向量，然后添加到训练矩阵中
        #trainMat.append(setOfWords2Vec(vocabList,fullTest[docIndex]))
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        #trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        #将该邮件的类标签存入训练类标签列表中
        trainClasses.append(classList[docIndex])

    #计算贝叶斯函数需要的概率值并返回
    p0V,p1V,pSpam = trainNB1(array(trainMat),array(trainClasses))
    errorCount = 0
    #遍历测试集中的字符串列表
    for docIndex in testSet:
        #同样将测试集中的字符串列表转为词条向量
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        #对测试集中字符串向量进行预测分类，分类结果不等于实际结果
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
    print('the error rate is: ', float(errorCount)/len(testSet))



'''
------------------ 示例：使用朴素贝叶斯分类器从个人广告中获取区域倾向 ------------------
1、收集数据：导入RSS源
>>> import feedparser  # 需 pip install feedparser 安装包
>>> ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
>>> ny['entries']
>>> len(ny['entries'])
2、源分类器及髙频词去除函数
>>> reload(bayes)
>>> ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
>>> sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
>>> vocabList,pSF,pNY=bayes.localWords(ny,sf)
>>> bayes.getTopWords(ny, sf)
'''
#实例：使用朴素贝叶斯分类器从个人广告中获取区域倾向
# RSS源分类器及高频词去除函数
def calMostFreq(vocabList,fullText):
    #导入操作符
    import operator
    #创建新的字典
    freqDict={}
    #遍历词条列表中的每一个词
    for token in vocabList:
        #将单词/单词出现的次数作为键值对存入字典
        freqDict[token] = fullText.count(token)
    #按照键值value(词条出现的次数)对字典进行排序，由大到小
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    #sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) 
    #返回出现次数最多的前30个单词
    return sortedFreq[:30]

def localWords(feed1, feed0):
    import feedparser
    #新建三个列表
    docList = [];  classList = []; fullText = []
    #获取条目较少的RSS源的条目数
    minLen = min(len(feed1['entries']),len(feed0['entries'])) #entries始终为空
    #遍历每一个条目
    for i in range(minLen):
        #解析和处理获取的相应数据
        wordList = textParse(feed1['entries'][i]['summary'])
        #添加词条列表到docList
        docList.append(wordList)
        #添加词条元素到 fullText
        fullText.extend(wordList)
        #类标签列表添加类1
        classList.append(1)
        #同上
        wordList = testParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        #此时添加类标签0
        classList.append(0)

    #构建出现的所有词条列表
    vocabList = createVocabList(docList)
    #找到出现的单词中频率最高的30个单词
    top30Words = calMostFreq(vocabList, fullText)
    #遍历每一个高频词，并将其在词条列表中移除
    #这里移除高频词后错误率下降，如果继续移除结构上的辅助词
    #错误率很可能会继续下降
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])

    #下面内容与函数spamTest完全相同
    # 训练数据集中数据的个数
    trainingSet = list(range(2 * minLen + 1)) #Python3
    #trainingSet = range(2*minLen);
    testSet=[]      # 测试数据集
    # 任意选择二十条数据作为测试数据
    for i in range(20):
        # 产生一个随机数
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex]) # 添加到测试集合上
        # 删除测试数据索引
        del(trainingSet[randIndex])

    trainMat=[];trainClasses=[]
    for docIndex in trainingSet:
        # 获得训练数据相应的词向量
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        # 添加相应的类别
        trainClasses.append(classList[docIndex])

    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount=0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount += 1 
    print('the error rate is:',float(errorCount)/len(testSet))
    return vocabList, p0V, p1V


# 按顺序输出 满足一定阈值词
#最具表征性的词汇显示函数
def getTopWords(ny,sf):
    import operator
    #利用RSS源分类器获取所有出现的词条列表，以及每个分类中每个单词出现的概率
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    #遍历每个类中各个单词的概率值
    for i in range(len(p0V)):
        #往相应元组列表中添加概率值大于阈值的单词及其概率值组成的二元列表
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))

    #对列表按照每个二元列表中的概率值项进行排序，排序规则由大到小
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**" * 14)
    #遍历列表中的每一个二元条目列表
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NF**" * 14)
    for item in sortedNY:
        print(item[0])


''' 主程序
'''
if __name__ == '__main__':
    # 训练函数trainNB1() 样例
    demo01()
    # beyes测试函数
    testingNB()
