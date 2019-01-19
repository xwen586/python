#!/usr/bin/env python3
# a1svm.py
"""
支持向量机，寻找最大间隔
SMO高效优化算法
参见: https://www.cnblogs.com/zy230530/p/6901277.html
"""
from numpy import *
from time import sleep

''' ------------------ SMO算法相关辅助中的辅助函数 ------------------ 
简易版SMO算法需要用到的一些功能
'''
# 数据文本读取和解析函数，返回数据与标签
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])]) # 数据
        labelMat.append(float(lineArr[2])) # 标签
    return dataMat, labelMat

# 随机选择函数：在样本集中随机选择不等于 i 的整数。m是所有alpha的数m。
def selectJrand(i, m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

# 约束范围函数：L<=alphaj<=H内的更新后的alphaj值
def clipAlpha(aj,H,L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj


''' ------------------ 简化版SMO算法 ------------------ 
#SMO算法的伪代码：
#创建一个alpha向量并将其初始化为0向量
#当迭代次数小于最大迭代次数时(w外循环)
    #对数据集中每个数据向量(内循环):
    #如果该数据向量可以被优化：
        #随机选择另外一个数据向量
        #同时优化这两个向量
        #如果两个向量都不能被优化，退出内循环
#如果所有向量都没有被优化，增加迭代次数，继续下一次循环
'''
#@dataMatIn ：数据集
#@classLabels：类别标签
#@C          ：权衡因子（增加松弛因子而在目标优化函数中引入了惩罚项）常数0
#@toler      ：容错率
#@maxIter    ：最大迭代次数
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn);
    labelMat = mat(classLabels).transpose() # 转置类别标签，得到列向量而不是列表
    b = 0; m,n = shape(dataMatrix)  # m,n-矩阵行列
    alphas = mat(zeros((m,1))) #新建一个m行1列的向量，初始值0
    iter = 0
    while (iter < maxIter):  #迭代循环
        alphaPairsChanged = 0  #记录alpha是否巳经进行优化，改变的alpha对数
        for i in range(m): #遍历样本集中的样本
            # 计算支持向量机算法的预测值，预测的类别
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            # 预测结果和真实结果进行比对，就可以计算误差Ei
            Ei = fXi - float(labelMat[i]) #if checks if an example violates KKT conditions
            # 如果误差很大，那么可以对该数据实例所对应的alpha值进行优化
            if(   (labelMat[i]*Ei < -toler) and (alphas[i] < C)) or \
                ( (labelMat[i]*Ei > toler)  and (alphas[i] > 0) ):
                j = selectJrand(i,m)  #随机选择第二个变量alphaj
                # 计算第二个变量对应数据的预测值 和 误差Ei
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                #记录alphai和alphaj的原始值，便于后续的比较
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                #如果两个alpha对应样本的标签不相同，求出相应的上下边界
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print("L==H"); continue
                #根据公式计算未经剪辑的alphaj
                #------------------------------------------
                # eta是 alphaJ 最优修改量
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                #如果eta>=0,跳出本次循环
                if eta >= 0: print("eta>=0"); continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                #------------------------------------------    
                #如果改变后的alphaj值变化不大，跳出本次循环    
                if (abs(alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); continue
                #否则，计算相应的alphai值
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                #再分别计算两个alpha情况下对于的b值
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                #如果0<alphai<C,那么b=b1
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2 #如果0<alphaj<C,那么b=b2
                else: b = (b1 + b2)/2.0 #否则，alphai，alphaj=0或C
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged) )
        #最后判断是否有改变的alpha对，没有就进行下一次迭代
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0  #否则，迭代次数置0，继续循环
        print("iteration number: %d" % iter)
    return b, alphas  #返回最后的b值和alpha向量

# 测试 简化版SMO
def demo01():
    dataArr, labelArr = loadDataSet(r'./ch06svm/testSet.txt')
    b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    print( b)
    print( alphas )

''' ------------------ 6.4 利用完整Platt SMO算法加速优化 ------------------ 
几百个点数据集，简化版SMO算法的运行是没有什么问题的，但是在更大数据集上的运行速度就会变慢。
完整版的Platt SMO算法应用了一些能够提速的启发方法。
'''
# 新建一个类的数据结构，保存当前重要的值
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        # 误差缓存，第一列给出的是eCache是否有效的标志位，第二列给出的是实际的E值。
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag

# 计算误差函数，方便多次调用
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b
    #fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

# 修改选择第二个变量alphaj的方法
def selectJ(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    #将误差矩阵每一行第一列置1，以此确定出误差不为0的样本
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

# 完整Platt SMO算法中的优化例程
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print("L==H"); return 0
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        #eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        #b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        #b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

# 完整Platt SMO算法中的外循环代码
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print( "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged) )
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged) )
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas

def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

# 测试 完整版SMO
def demo02():
    dataArr, labelArr = loadDataSet(r'./ch06svm/testSet.txt')
    b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    print('b=', b)
    #print( alphas )
    ws = calcWs(alphas,dataArr,labelArr)
    print('ws=', ws)



''' 主程序
'''
if __name__ == '__main__':
    # 简化版SMO
    #demo01()
    # 完整版SMO
    demo02()
