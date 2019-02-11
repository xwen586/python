#-*- coding:utf-8 -*-
# d01matr.py
""" 矩阵基础 """
import numpy as np

''' 创建矩阵 '''
a = np.mat("3,-1; 2,0")
b = np.mat([[-7,2], [-3,5]])
cA = np.array([0,8,-9])  #数组转换
dA = np.arange(1,9).reshape(-1,3)  #数组转换
c = np.mat(cA)
d = np.mat(dA)

#单位矩阵
e = np.eye(3)

# 全0矩阵和全1的矩阵
mZ = np.zeros([2,3])#3*5的全0矩阵
m1 = np.ones([2,3])##3*5的全1矩阵
print(mZ)
print(m1)

#生成随机矩阵
mR = np.random.rand(2,3)#2行3列的0~1之间的随机数矩阵
print(mR)


''' 矩阵的元素运算 '''
# 元素求和
mymat = np.mat([[1,2,3],[4,5,6],[7,8,9]])
print(mymat.sum())  #矩阵所有元素求和
print(sum(mymat))   #默认是按列求和，得到[[12 15 18]]。python的sum() 与 np.sum() 区别
print(sum(dA))   #得到[12 15 18]。
print(np.sum(mymat)) #默认所有元素求和，可指定列求和axis=0；行求和axis=1

#矩阵各元素的n次幂:n=2
print( np.power(mymat, 2))


'''Linalg线性代数库
.矩阵的行列式
.矩阵的逆
.矩阵的对称
.矩阵的秩
.可逆矩阵求解线性方程
'''



