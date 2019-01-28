#!/usr/bin/env python3
# a1pca.py
""" 降维技术-PCA 主成分分析

"""
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    #datArr = [map(float,line) for line in stringArr] #Python3.x的map类型变动
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)

''' --------------PCA --------------
伪码大致如下：
去除平均值
计算协方差矩阵
计算协方差矩阵的特征值和特征向量
将特征值从大到小排序
保留最上面的#个特征向量
将数据转换到上述#个特征向量构建的新空间中
'''
def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0) #求均值
    meanRemoved = dataMat - meanVals #remove mean #归一化数据
    covMat = cov(meanRemoved, rowvar=0) #求协方差
    eigVals,eigVects = linalg.eig(mat(covMat)) #计算特征值和特征向量
    #对特征值进行排序，默认从小到大
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions #逆序取得特征值最大的元素
    redEigVects = eigVects[:,eigValInd] #用特征向量构成矩阵 #reorganize eig vects largest to smallest
    #用归一化后的各个数据与特征矩阵相乘，映射到新的空间
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

#在NumPy中实现PCA
def demo01():
    dataMat = loadDataSet(r'./ch13dimred/data/testSet.txt')
    lowDMat, reconMat = pca(dataMat, 1)
    print('topNfeat=1,shape:',shape(lowDMat))
    #没有剔除任何特征，那么重构之后的数据会和原始的数据重合。
    lowDMat2, reconMat2 = pca(dataMat, 2)
    print('topNfeat=2,shape:',shape(lowDMat2))

    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    fig, ax = plt.subplots(1,2, figsize=(8,4))
    print('ax=', len(ax), '；shape=', shape(ax))
    ax[0].scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0], marker='^', s=10, c='blue')
    ax[0].scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0], marker='o', s=4, c='red')
    #ax[1].scatter(xSort[:,1], y2Hat[srtInd], color='blue');
    #ax[1].scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
    ax[1].scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0], marker='^', s=10, c='blue')
    ax[1].scatter(reconMat2[:,0].flatten().A[0], reconMat2[:,1].flatten().A[0], marker='o', s=4, c='red')
    plt.tight_layout()  #自动调整子插图
    plt.show()


''' --------------示例：利用PCA对半导体制造数据降维--------------
半导体制造过程中存在瑕疵，通过早期测试和频繁测试来发现有缺陷的产品。
如果机器学习技术能够用于进一步减少错误。
考察制造过程数据集，拥有590个特征，对这些特征进行降维处理。
'''
# 所有样本都有NAN（缺失值），用平均值来代替缺失值。
def replaceNanWithMean(): 
    datMat = loadDataSet(r'./ch13dimred/data/secom.data', ' ')
    numFeat = shape(datMat)[1]  #获取特征维度，即列的数目
    #print("特征维度:", numFeat)
    for i in range(numFeat):
        #计算均值（以列为单位计算）
        #~isnan返回boolean矩阵,其中不是NAN的位置标为true,是NAN的标为false
        #nonzero返回元组,其中0处为非0(True为非0)元素行标,1处存放0元素行标
        meanVal = mean(datMat[nonzero( ~isnan(datMat[:,i].A) )[0],i]) #values that are not NaN (a number)
        # 将该维度中所有NaN特征全部用均值替换
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat

def demo02():
    dataMat = replaceNanWithMean()
    print("特征维度:", shape(dataMat)[1])
    meanVals = mean(dataMat, axis=0)  # 对各列求均值
    meanRemoved = dataMat - meanVals  # 各列求差，减去平均值
    covMat = cov(meanRemoved, rowvar=0)  # 计算协方差矩阵
    eigVals,eigVects = linalg.eig(mat(covMat))  # 计算协方差矩阵的特征值 和特征向量
    eigVer = eigVals/sum(eigVals)
    print('90%的主成分方差总和:', sum(eigVals)*0.9)     # 计算90%的主成分方差总和
    print('前6个主成分所占的方差:', sum(eigVals[:6]))     # 计算前6个主成分所占的方差
    for i in arange(0,7):
        print("%d,  %f,\t %f,\t %f" % (i+1, eigVer[i]*100, sum(eigVer[:i+1])*100, eigVects[i,i]) )
    lowDMat1, reconMat1= pca(dataMat, 1)# 降维成1维矩阵前一个主成份 
    lowDMat2, reconMat2= pca(dataMat, 2)# 降维成2维矩阵前两个主成份 
    lowDMat3, reconMat3= pca(dataMat, 3)# 降维成3维矩阵前三个主成份 
    lowDMat6, reconMat6= pca(dataMat, 6)# 降维成6维矩阵前六个主成份 
    print("原数据维度: ", shape(dataMat))
    print("降维后数据维度1: ", shape(lowDMat1))
    print("降维后数据维度2: ", shape(lowDMat2))
    print("降维后数据维度3: ", shape(lowDMat3)) 
    print("降维后数据维度6: ", shape(lowDMat6))
    plt.plot(eigVals[:20])      # 对前20个画图观察
    plt.scatter(arange(0,20),eigVals[:20],color='blue',marker='o')
    plt.show()


if __name__ == '__main__':
	#
    #demo01()
	#
    demo02()
