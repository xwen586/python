#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# a02sklearn.py
"""scikit-learn 随机数据生成介绍
https://blog.csdn.net/weixin_42039090/article/details/80614918
scikit-learn生成随机数据的API都在datasets类之中，适合特定机器学习模型的数据。
常用的API有：
 1) 回归模型：用make_regression 生成回归模型的数据
 2) 分类模型：用make_hastie_10_2，make_classification或者
    make_multilabel_classification生成分类模型数据
 3) 聚类模型：用make_blobs生成聚类模型数据
 4) 正态分布：用make_gaussian_quantiles生成分组多维正态分布的数据
"""
import numpy as np
import matplotlib.pyplot as plt


class a02sklearn(object):
    def __init__(self): #构造器
        print('initialized')
        #pass
    def __del__(self): # "destructor" 解构器
        print('destructor')

    def Run(self):
        print("Hello class!")
        self.regression() # 回归模型随机数据
        self.classification() # 分类模型随机数据
        self.blobs()  # 聚类模型随机数据
        self.quantiles()  # 分组正态分布混合数据

    # 回归模型随机数据
    def regression(self):
        from sklearn.datasets.samples_generator import make_regression
        # X为样本特征，y为样本输出， coef为回归系数，共200个样本，每个样本1个特征
        X, Y, coef =make_regression(n_samples=200, n_features=1, noise=20, coef=True)
        # 画图
        plt.scatter(X, Y, color='orange')
        plt.plot(X, X*coef, color='blue', linewidth=2)
        plt.xticks(())
        plt.yticks(())
        plt.show()

    # 分类模型随机数据
    def classification(self):
        from sklearn.datasets.samples_generator import make_classification
        # X1为样本特征，Y1为样本类别输出， 共400个样本，每个样本2个特征，输出有3个类别，没有冗余特征，每个类别一个簇
        X1, Y1 = make_classification(n_samples=400, n_features=2, n_redundant=0,
                                     n_clusters_per_class=1, n_classes=3)
        plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
        plt.show()

    # 聚类模型随机数据
    def blobs(self):
        from sklearn.datasets.samples_generator import make_blobs
        # X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共3个簇，簇中心在[-1,-1], [1,1], [2,2]， 簇方差分别为[0.4, 0.5, 0.2]
        X, Y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [1,1], [2,2]], cluster_std=[0.4, 0.5, 0.2])
        plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)
        plt.show()

    # 分组正态分布混合数据
    def quantiles(self):
        from sklearn.datasets import make_gaussian_quantiles
        #生成2维正态分布，生成的数据按分位数分成3组，1000个样本,2个样本特征均值为1和2，协方差系数为2
        X1, Y1 = make_gaussian_quantiles(n_samples=1000, n_features=2, n_classes=3, mean=[1,2],cov=2)
        plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
        plt.show()



if __name__ == '__main__':
    a = a02sklearn()
    a.Run()
    #del a
