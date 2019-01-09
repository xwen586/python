#!/usr/bin/env python3
# a01check.py
"""检查与准备"""

''' ---------- 访问数据 ---------- 
1. 通过REST 风格的API 接口
2. 库是Python Request
'''
import requests
r = requests.get(r"https://api.github.com/users/acombs/starred")
r.json()

''' ---------- Pandas ---------- 
一个卓越的数据分析工具
'''
import os
import pandas as pd
import requests

#PATH = r'/Users/alexcombs/Desktop/iris/'
PATH = r'.\ch01prep'
# 获取 鸢尾花 Iris 数据集
r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
with open(PATH + r'\iris.data', 'w') as f:
    f.write(r.text)

os.chdir(PATH)  #PATH + r
# 读取数据，并添加标题行
# sepal length/width-花萼长/宽
# petal length/width-花瓣长/宽
df = pd.read_csv('iris.data', names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
df.head(5)  # 查看前5行数据
df.tail(5)  # 查看最后5行的数据
df.count()
len(df)    #获取数据的行数

# 只包含Iris-virginica 类的数据。
df[df['class']=='Iris-virginica']
df[df['class']=='Iris-virginica'].count() #数量

virginica = df[df['class']=='Iris-virginica'].reset_index(drop=True)

#只包含来自Iris-virginica 类、而且花瓣宽度大于2.2 的数据。
df[(df['class']=='Iris-virginica')&(df['petal width']>2.2)]

# 使用Pandas，从虹膜数据集中获取描述性统计数据
df.describe()

# 传入自定义的百分比
df.describe(percentiles=[.20,.40,.80,.90,.95])

# 特征之间是否有任何相关性
df.corr()

# df.iloc(20) # 仅当有数字索引
