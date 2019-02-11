#-*- coding:utf-8 -*-
# d01norm.py
""" 范数
闵可夫斯基距离(Minkowski Distance)
常用的向量的范数：
L1范数:  ||x|| 为x向量各个元素绝对值之和。
L2范数:  ||x||为x向量各个元素平方和的1/2次方，L2范数又称Euclidean范数或者Frobenius范数
Lp范数:  ||x||为x向量各个元素绝对值p次方和的1/p次方
L∞范数:  ||x||为x向量各个元素绝对值最大那个元素的绝对值
https://blog.csdn.net/liukuan73/article/details/80494779
"""
import numpy as np 
import numpy.linalg as la

# 范数示例
A = np.array([0, 8, -9])
Amod = np.sqrt(sum(np.power(A,2)))
Anorm= la.norm(A)
print("modA:", Amod, "\nnorm(A):", Anorm)

#曼哈顿距离(Manhattan Distance)：对应L1-范数
v1 = np.array([1,2,3])  # matrix的值会不同
v2 = np.array([4,5,6])
print(v1-v2)
md = np.sum(np.abs(v1-v2))
vnorm= la.norm(v1-v2, ord=1)
print("md:", md, "vnorm:", vnorm)


#欧氏距离(Euclidean Distance)：对应L2-范数
X0 = np.matrix('1, 2, 3')  #('1 2 3; 4 5 6; 7 8 9')
X1 = np.matrix('4, 5, 6')  #('0 1 1; 1 0 1; 1 1 0')
xed = np.sqrt(sum(X0-X1)*(X0-X1).T)
xnorm= la.norm(X0-X1, 2)
print("xed:", xed, "xnorm:", xnorm)


#4.切比雪夫距离(Chebyshev Distance)
v1 = np.array([1,2,3])  # matrix的值会不同
v2 = np.array([4,5,6])
cd = np.abs(v1-v2).max()
vnorm= la.norm(v1-v2, ord=np.inf)
print("cd:", cd, "vnorm:", vnorm)

