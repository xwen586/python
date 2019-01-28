#!/usr/bin/env python3
# b1pca.py
""" 降维技术-PCA
使用sklearn中PCA处理方法
"""
from sklearn.decomposition import PCA
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    #datArr = [map(float,line) for line in stringArr] #Python3.x的map类型变动
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)

''' --------------示例：利用PCA对半导体制造数据降维--------------
sklearn.decomposition.PCA(n_components=None, copy=True, whiten=False)
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

# 与a1pca.py中的demo02()结果一致
def demo01():
    dataMat = replaceNanWithMean()
    pca = PCA()  # n_components参数选择降维程度;默认为None，特征个数不会改变（特征数据本身会改变）
    pca = pca.fit(dataMat) # fit_transform()为转换数据
    main_var = pca.explained_variance_  # 特征值 同eigVals,eigVects = linalg.eig()
    print("特征维度:", shape(dataMat)[1])
    print('90%的主成分方差总和:', sum(main_var)*0.9)  # 计算90%的主成分方差总和
    print('前6个主成分所占的方差:', sum(main_var[:6]))    # 计算前6个主成分所占的方差
    eigVer = main_var/sum(main_var)
    pcavar = pca.explained_variance_ratio_
    for i in arange(0,7):
        print("%d,  %f,\t %f" % (i+1, eigVer[i]*100, sum(eigVer[:i+1])*100) )
    plt.plot(main_var[:20])     # 对前20个画图观察
    plt.scatter(arange(0,20),main_var[:20],color='blue',marker='o')
    plt.show()

# 降维
def demo02():
    dataMat = replaceNanWithMean()
    #lowDMat1, reconMat1= pca(dataMat, 1)# 降维成1维矩阵前一个主成份
    #lowDMat2, reconMat2= pca(dataMat, 2)# 降维成2维矩阵前两个主成份
    #lowDMat3, reconMat3= pca(dataMat, 3)# 降维成3维矩阵前三个主成份
    #lowDMat6, reconMat6= pca(dataMat, 6)# 降维成6维矩阵前六个主成份
    pca = PCA(n_components=1)
    lowDMat1 = pca.fit_transform(dataMat)
    pcavar1 = pca.explained_variance_ratio_
    pca = PCA(n_components=2)
    lowDMat2 = pca.fit_transform(dataMat)
    pcavar2 = pca.explained_variance_ratio_
    pca = PCA(n_components=3)
    lowDMat3 = pca.fit_transform(dataMat)
    pcavar3 = pca.explained_variance_ratio_
    pca = PCA(n_components=6)
    lowDMat6 = pca.fit_transform(dataMat)
    pcavar6 = pca.explained_variance_ratio_

    print("原数据维度: ", shape(dataMat))
    print("降维后数据维度1:%s 方差百分比:[%f]" % (str(shape(lowDMat1)), pcavar1))
    print("降维后数据维度2:%s" % str(shape(lowDMat2)), "方差百分比:", pcavar2)
    print("降维后数据维度3:%s" % str(shape(lowDMat3)), "方差百分比:", pcavar3)
    print("降维后数据维度6:%s" % str(shape(lowDMat6)), "方差百分比:", pcavar6)


'''------------- 三维的数据来降维 -------------
#https://www.cnblogs.com/pinard/p/6243025.html

'''
def demo03():
    #import numpy as np
    #import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.datasets.samples_generator import make_blobs
    # X为样本特征，Y为样本簇类别， 共1000个样本，每个样本3个特征，共4个簇
    X, y = make_blobs(n_samples=10000, n_features=3, \
        centers=[[3,3,3], [0,0,0], [1,1,1], [2,2,2]],\
        cluster_std=[0.2, 0.1, 0.2, 0.2], \
        random_state =9)
    fig = plt.figure()
    ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
    plt.scatter(X[:,0], X[:,1], X[:,2], marker='o')
    plt.show()

    #先不降维，只对数据进行投影，看看投影后的三个维度的方差分布
    #from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)

    #降维，从三维降到2维
    pca = PCA(n_components=2)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    #转化后的数据分布,降维后的数据依然可以很清楚的看到我们之前三维图中的4个簇。
    Xnew = pca.transform(X)
    plt.scatter(Xnew[:,0], Xnew[:,1], marker='o')
    plt.show()

    #不直接指定降维的维度，而指定降维后的主成分方差和比例。
    pca = PCA(n_components=0.95)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    print( pca.explained_variance_)
    print( pca.n_components_)
    #现在选择阈值99%看看，
    pca = PCA(n_components=0.99)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    print( pca.explained_variance_)
    print( pca.n_components_)
    #让MLE算法自己选择降维维度的效果
    pca = PCA(n_components='mle', svd_solver='full')
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    print( pca.explained_variance_)
    print( pca.n_components_)



if __name__ == '__main__':
	#
    demo01()
	#
    demo02()
	#三维数据的降维
    demo03()

