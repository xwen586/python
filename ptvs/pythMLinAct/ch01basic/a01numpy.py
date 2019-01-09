#!/usr/bin/env python3
# a01numpy.py
"""numpy基础"""
from numpy import *

rand = random.rand(4,4)  # 构造了一个4x4的随机数组 array
randMat = mat(random.rand(4,4))  # 为矩阵 matrix

invRandMat = randMat.I  # 矩阵求逆

myEye = randMat * invRandMat  # 应得到单位矩阵，对角线元素是1，其他为0
e = eye(4) #创建4x4的单位矩阵

